"""Loads `opencl_tf_ops.so` and points it at the bundled `.cl` files."""

from __future__ import annotations

import os
from pathlib import Path

import tensorflow as tf

_HERE     = Path(__file__).resolve().parent   # .../opencl_tf/
_LIB_PATH = _HERE / "opencl_tf_ops.so"

# Kernel search order:
#   1. Bundled inside the package (pip-installed copy):  opencl_tf/kernels/
#   2. Repo root layout (editable / dev checkout):       opencl_tf/../kernels/
# The C++ CLBackend also honours the OPENCL_TF_KERNELS_PATH env var (step 2
# in its own search order), so setdefault here lets users override if needed.
_KERNEL_DIR_BUNDLED = _HERE / "kernels"
_KERNEL_DIR_DEV     = _HERE.parent / "kernels"
_KERNEL_DIR = (
    _KERNEL_DIR_BUNDLED if _KERNEL_DIR_BUNDLED.is_dir() else _KERNEL_DIR_DEV
)
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
