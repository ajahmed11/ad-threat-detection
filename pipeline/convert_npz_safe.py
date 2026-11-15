"""One-off migration: make an OLD windows.npz loadable with allow_pickle=False.

Older builds stored the attack-type column as a pickled object array. Rebuilding
with pipeline/build_dataset.py produces the safe format directly; use this only
to convert a dataset YOU created and trust (it loads the legacy file with
pickle enabled, exactly once).

    python pipeline/convert_npz_safe.py path/to/windows.npz
"""
import sys
from pathlib import Path

import numpy as np


def main(path: str) -> int:
    src = Path(path)
    legacy = np.load(src, allow_pickle=True)          # trusted, locally-built file only
    out = {}
    for k in legacy.files:
        a = legacy[k]
        out[k] = a.astype(str) if a.dtype == object else a
    dst = src.with_name(src.stem + "_safe.npz")
    np.savez_compressed(dst, **out)
    np.load(dst, allow_pickle=False).files            # proves it now loads safely
    print(f"wrote {dst}  (back up the original, then rename to {src.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
