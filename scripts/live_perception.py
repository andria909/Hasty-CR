"""Board-only, multi-scale inference around the vendored ONNX detector.

Keep fixes in the project so re-vendoring does not lose them. No weights or
confidence thresholds of the original full-frame pass are changed.
"""

import numpy as np


def health_bar_side(image, bbox):
    """A clear horizontal health bar can correct the 16px side classifier.

    Use thin horizontal components rather than overall sprite colour.
    Abstain on mixed bars, short marks and occlusions.
    """
    left, top, right, bottom = bbox
    width, height = right-left, bottom-top
    crop = image.crop((max(0, left-3), max(0, top-height*.6),
                       min(image.width, right+3), min(image.height, top+height*.4)))
    rgb = np.asarray(crop.convert("RGB"), dtype=np.int16)
    if not rgb.size:
        return None
    red = (rgb[..., 0] > 140) & (rgb[..., 0] > rgb[..., 1]*1.3) & (rgb[..., 0] > rgb[..., 2]*1.3)
    blue = (rgb[..., 2] > 140) & (rgb[..., 2] > rgb[..., 0]*1.3) & (rgb[..., 2] > rgb[..., 1]*1.05)
    def longest(mask):
        from scipy.ndimage import find_objects, label
        components, _ = label(mask)
        sizes = [(box[1].stop-box[1].start, box[0].stop-box[0].start)
                 for box in find_objects(components) if box is not None]
        return max((w for w, h in sizes if w >= 8 and w >= h*3
                    and h <= max(4, image.height*.015)), default=0)
    a, e = longest(blue), longest(red)
    if max(a, e) < max(8, width*.4) or max(a, e) < 1.8 * max(1, min(a, e)):
        return None
    return "ally" if a > e else "enemy"


def box_iou(a, b):
    width = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    area_a = max(0., a[2]-a[0]) * max(0., a[3]-a[1])
    area_b = max(0., b[2]-b[0]) * max(0., b[3]-b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def merge_board_predictions(predictions, width, height, building_ids=frozenset()):
    """Remove scenery and duplicate hypotheses, before assigning a side.

    The bottom-centre is the unit's ground anchor. Borders and the king-level
    badge below the arena are not playable ground. Do not mask king/princess
    towers: real troops can walk beside and behind them.
    """
    kept = []
    for row in sorted(predictions, key=lambda row: float(row[4]), reverse=True):
        left, top, right, bottom, confidence, cls = row
        x = (left + right) / (2 * width)
        y = bottom / height
        if right <= left or bottom <= top or not (.07 <= x <= .93 and .05 <= y <= .755):
            continue
        # Different labels for effectively the identical box are competing
        # hypotheses, not an extra unit. Less-overlapping troops survive.
        if any(box_iou(row, other) > (.5 if
                   (cls in building_ids and other[5] in building_ids) else .9)
               for other in kept):
            continue
        kept.append(row)
    return np.asarray(kept, dtype=np.float32).reshape(-1, 6)


class BoardUnitDetector:
    """Original view plus bounded building proposals at a larger scale."""

    ROI_MIN_CONF = .65
    MAX_REFINEMENTS = 2

    def __init__(self, upstream):
        self.upstream = upstream

    def run(self, image):
        from clashroyalebuildabot.constants import DETECTOR_UNITS
        u = self.upstream
        width, height = image.size
        pixels, padding = u._preprocess(image)
        rows = u._infer(pixels)[0]
        rows = rows[rows[:, 4] > .15].copy()
        rows = u.fix_bboxes(rows, width, height, padding)
        rows[:, [1, 3]] *= u.UNIT_Y_END - u.UNIT_Y_START
        rows[:, [1, 3]] += u.UNIT_Y_START * height
        predictions = list(rows[rows[:, 4] > u.MIN_CONF])
        building_ids = {i for i, unit in enumerate(DETECTOR_UNITS)
                        if unit.category.name == "BUILDING"}
        # Retain weak building proposals as locations, never as accepted units.
        # A second view must independently identify the object above .65.
        proposals = merge_board_predictions(
            [row for row in rows if row[5] in building_ids], width, height, building_ids)
        refinements = 0
        for proposal in proposals:
            if proposal[4] >= self.ROI_MIN_CONF:
                continue
            if refinements >= self.MAX_REFINEMENTS:
                break
            refinements += 1
            margin = width * .19
            box = (max(0, round(proposal[0]-margin)), max(0, round(proposal[1]-margin)),
                   min(width, round(proposal[2]+margin)), min(height, round(proposal[3]+margin)))
            crop = image.crop(box)
            pixels, padding = u.resize_pad_transpose_and_scale(crop)
            rows = u._infer(np.expand_dims(pixels, 0))[0]
            rows = rows[rows[:, 4] > self.ROI_MIN_CONF].copy()
            # Extra views are for missed buildings. Letting every extra troop
            # hypothesis into the state introduces scale-dependent ghosts.
            rows = rows[[DETECTOR_UNITS[int(row[5])].category.name == "BUILDING"
                         for row in rows]] if len(rows) else rows
            rows = u.fix_bboxes(rows, crop.width, crop.height, padding)
            rows[:, [0, 2]] += box[0]
            rows[:, [1, 3]] += box[1]
            predictions.extend(row for row in rows if box_iou(row, proposal) > .25)
        merged = merge_board_predictions(predictions, width, height, building_ids)
        # Upstream post-processing expects y in pre-crop coordinates and applies
        # the crop transform. Undo it here exactly once; x is already global.
        merged[:, [1, 3]] -= u.UNIT_Y_START * height
        merged[:, [1, 3]] /= u.UNIT_Y_END - u.UNIT_Y_START
        allies, enemies = u._post_process(merged, height, image)
        corrected = ([], [])
        for original, detections in (("ally", allies), ("enemy", enemies)):
            for detection in detections:
                side = None
                if (detection.unit.name in u.possible_ally_names
                        and detection.unit.category.name == "BUILDING"):
                    side = health_bar_side(image, detection.position.bbox)
                corrected[0 if (side or original) == "ally" else 1].append(detection)
        return corrected


class EmptyUnitDetector:
    def run(self, image):
        return [], []


def configure_unit_detection(detector, vision):
    """Use board corrections for ONNX, or skip it when YOLO is ready."""
    if vision is not None and vision.ready:
        detector.unit_detector = EmptyUnitDetector()
    elif hasattr(detector.unit_detector, "_infer") and not isinstance(detector.unit_detector, BoardUnitDetector):
        detector.unit_detector = BoardUnitDetector(detector.unit_detector)
