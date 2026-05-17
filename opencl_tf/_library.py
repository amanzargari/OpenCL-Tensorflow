"""Loads `opencl_tf_ops.so` and points it at the bundled `.cl` files."""

from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf

_HERE       = Path(__file__).resolve().parent          # .../opencl_tf
_LIB_PATH   = _HERE / "opencl_tf_ops.so"
_KERNEL_DIR = (_HERE.parent / "kernels").resolve()      # .../kernels

# Tell the .so where kernel source files live. Setdefault so users can
# override (e.g. when running an installed copy from a different layout).
os.environ.setdefault("OPENCL_TF_KERNELS_PATH", str(_KERNEL_DIR))

if not _LIB_PATH.exists():
    raise ImportError(
        f"opencl_tf_ops.so not found at {_LIB_PATH}.\n"
        f"Build it first:\n"
        f"    cmake -S . -B build && cmake --build build -j\n"
        f"or:\n"
        f"    make"
    )

raw_ops = tf.load_op_library(str(_LIB_PATH))
"""The raw `tf.load_op_library` handle. Prefer the wrappers in
`opencl_tf.ops.conv2d` over reaching for raw ops directly."""
