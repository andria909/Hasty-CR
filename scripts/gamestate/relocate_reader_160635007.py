"""Relocate our existing reader's root constant; never modifies the game.

Use when the Android cross compiler is unavailable. Requires the exact v9
reader and changes only its MOV immediate. Source builds use
-DHASTY_ROOT_OFFSET=0x1b7add0 with native/arm64_entities.c instead.
"""
import argparse
import hashlib
from pathlib import Path
import struct

SOURCE_SHA = '6c7abccf63656b3f273aa049611a07868c74a22d124dd899063385992e976a34'
OUTPUT_SHA = 'edcb1402b028761b816b217dacf8c2555538847d365913a62aac677a65d050a8'


def relocate(blob: bytes) -> bytes:
    if hashlib.sha256(blob).hexdigest() != SOURCE_SHA:
        raise ValueError('Expected exact v9 reader SHA256')
    # MOV w8,#0xc210; MOVK w8,#0x1b7,LSL#16; ADD x8,x23,x8.
    pattern = struct.pack('<III', 0x52984208, 0x72a036e8, 0x8b0802e8)
    if blob.count(pattern) != 1 or blob.index(pattern) != 10228:
        raise ValueError('Root-load instruction sequence differs')
    result = bytearray(blob)
    struct.pack_into('<I', result, 10228, 0x52800008 | (0xadd0 << 5))
    if hashlib.sha256(result).hexdigest() != OUTPUT_SHA:
        raise ValueError('Relocated reader SHA256 differs')
    return bytes(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    result = relocate(a.source.read_bytes())
    with a.output.open('xb') as stream:
        stream.write(result)
    a.output.chmod(0o755)
    print(OUTPUT_SHA)


if __name__ == '__main__':
    main()
