"""Offline adapter for the published KataCR v0.7.13 detector pair.

The last sigmoid channel is team, NOT an object class. Training-only classes
in the original pickle are mapped to their Ultralytics base classes; forward
inference is unchanged. No import of KataCR's training stack is needed.
"""
from __future__ import annotations

import pickle
import time
import types
from pathlib import Path

import numpy as np
from PIL import Image

from .vision import NOT_A_UNIT, NOT_A_UNIT_EITHER


def load_checkpoint(path: Path):
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss
    from ultralytics.utils.tal import TaskAlignedAssigner

    class Reader(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "katacr.yolov8.custom_model":
                return {"CRDetectionModel": DetectionModel,
                        "CRDetectionLoss": v8DetectionLoss,
                        "CRTaskAlignedAssigner": TaskAlignedAssigner}[name]
            return super().find_class(module, name)

    compat = types.ModuleType("katacr_checkpoint_pickle")
    compat.Unpickler = Reader
    compat.load = pickle.load
    checkpoint = torch.load(path, map_location="cpu", weights_only=False, pickle_module=compat)
    model = (checkpoint.get("ema") or checkpoint["model"]).float().eval()
    if model.names.get(model.model[-1].nc-1) != "padding_belong":
        raise ValueError("Expected KataCR's final team channel (padding_belong)")
    return model


def decode_predictions(output, names, confidence=.4, iou=.6):
    """Decode xywh + classes + team; return CPU xyxy/conf/class/team-probability."""
    import torch
    from torchvision.ops import nms
    output = output[0] if isinstance(output, tuple) else output
    if output.shape[0] != 1 or output.shape[1] != 4+len(names):
        raise ValueError("Unexpected KataCR output shape")
    values = output[0].detach().float().cpu().T
    scores, classes = values[:, 4:-1].max(1)
    keep = scores > confidence
    values, scores, classes = values[keep], scores[keep], classes[keep]
    if not len(values):
        return torch.empty((0, 7))
    xywh = values[:, :4]
    boxes = torch.cat((xywh[:, :2]-xywh[:, 2:]/2, xywh[:, :2]+xywh[:, 2:]/2), 1)
    indices = nms(boxes + classes[:, None]*7680, scores, iou)[:300]
    return torch.cat((boxes, scores[:, None], classes[:, None], values[:, -1:]), 1)[indices]


def unit_name(raw):
    base = raw.removesuffix("-evolution").removesuffix("-hero")
    non_units = {"giant-snowball", "goblin-curse", "void", "vines", "royal-chef", "skeleton-king-skill"}
    if (base in NOT_A_UNIT or base in NOT_A_UNIT_EITHER or base in non_units or raw.startswith("padding_")
            or raw.endswith("-symbol")):
        return None
    if raw.endswith("-evolution"):
        raw = raw.removesuffix("-evolution")
    return {"mini-pekka": "minipekka", "phoenix-big": "phoenix_large",
            "phoenix-small": "phoenix_small", "elixir-golem-big": "elixir_golem_large",
            "elixir-golem-mid": "elixir_golem_medium"}.get(raw, raw.replace("-", "_"))


class KataCRDetector:
    def __init__(self, weights, device="cpu", confidence=.4, crop=(.02, 0., .98, .81)):
        import torch
        self.device = device
        self.confidence = confidence
        # Initial MuMu-specific crop. Explicit in reports; not a universal arena calibration.
        self.crop = crop
        self.models = [load_checkpoint(Path(path)).fuse(verbose=False).to(device) for path in weights]
        self.last_ms = 0.
        self.torch = torch

    def detect(self, frame):
        from torchvision.ops import nms
        torch = self.torch
        start = time.perf_counter()
        w, h = frame.size
        left, top, right, bottom = [round(v*s) for v, s in zip(self.crop, (w,h)*2)]
        # Published arena images are 568x896. Pad to stride 32 after resizing.
        crop = frame.convert("RGB").crop((left,top,right,bottom)).resize((568,896), Image.Resampling.BILINEAR)
        canvas = np.full((896,576,3), 114, dtype=np.uint8)
        canvas[:, 4:572] = np.asarray(crop)
        pixels = torch.from_numpy(canvas.transpose(2,0,1).copy()).float()[None].to(self.device)/255
        candidates = []
        with torch.inference_mode():
            for model in self.models:
                decoded = decode_predictions(model(pixels), model.names, self.confidence)
                for row in decoded.tolist():
                    x1,y1,x2,y2,score,cls,probability = row
                    candidates.append({"raw_name": model.names[int(cls)], "confidence": score,
                                       "team_probability": probability,
                                       "bbox": [(x1-4)*(right-left)/568+left,
                                                y1*(bottom-top)/896+top,
                                                (x2-4)*(right-left)/568+left,
                                                y2*(bottom-top)/896+top]})
        if candidates:
            boxes = torch.tensor([row["bbox"] for row in candidates])
            scores = torch.tensor([row["confidence"] for row in candidates])
            # Match author's class-agnostic merge across the two scale specialists.
            candidates = [candidates[i] for i in nms(boxes, scores, .6).tolist()]
        detections = []
        for row in candidates:
            name = unit_name(row["raw_name"])
            if name is None:
                continue
            probability = row["team_probability"]
            detections.append({**row, "name": name,
                               "side": "enemy" if probability > .5 else "ally",
                               "side_confidence": max(probability, 1-probability)})
        if self.device == "mps":
            torch.mps.synchronize()
        self.last_ms = (time.perf_counter()-start)*1000
        return detections
