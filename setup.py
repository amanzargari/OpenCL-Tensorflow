"""
Build the opencl_tf_ops.so shared library via CMake and bundle it with
the Python package.

Usage
-----
    pip install .                           # build + install
    pip install -e .                        # editable (dev) install
    python setup.py build_ext --inplace     # compile .so into source tree only

System requirements
-------------------
    cmake >= 3.18
    opencl-headers   (sudo apt install opencl-headers)
    ocl-icd-opencl-dev   (sudo apt install ocl-icd-opencl-dev)
    tensorflow in the active Python environment
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeBuildExt(build_ext):
    """Compile opencl_tf_ops.so via CMake instead of the normal distutils flow."""

    def build_extensions(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):  # noqa: ARG002
        source_dir = Path(__file__).parent.resolve()
        cmake_build_dir = Path(self.build_temp) / "cmake"
        cmake_build_dir.mkdir(parents=True, exist_ok=True)

        # Destination for the compiled .so (and bundled kernels).
        # --inplace (editable/dev): compile directly into the source package dir.
        # Regular wheel/install:    stage into build_lib for packaging.
        if self.inplace:
            pkg_out = source_dir / "opencl_tf"
        else:
            pkg_out = Path(self.build_lib) / "opencl_tf"
        pkg_out.mkdir(parents=True, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={pkg_out}",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        build_args = [f"-j{os.cpu_count() or 1}"]

        try:
            subprocess.check_call(
                ["cmake", str(source_dir)] + cmake_args,
                cwd=cmake_build_dir,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "cmake not found. Install it with:\n"
                "  sudo apt install cmake          # Debian / Ubuntu\n"
                "  brew install cmake              # macOS\n"
                "  conda install -c conda-forge cmake"
            ) from None

        try:
            subprocess.check_call(
                ["cmake", "--build", str(cmake_build_dir)] + build_args,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"CMake build failed (exit {exc.returncode}).\n"
                "Check that the OpenCL development headers are installed:\n"
                "  sudo apt install opencl-headers ocl-icd-opencl-dev\n"
                "and that TensorFlow is importable in the current environment."
            ) from exc

        # For non-editable installs, bundle the .cl kernel files inside the
        # package so the installed copy is self-contained.
        if not self.inplace:
            shutil.copytree(
                source_dir / "kernels",
                pkg_out / "kernels",
                dirs_exist_ok=True,
            )

    def get_outputs(self):
        # CMake places the .so directly into pkg_out; the normal distutils
        # copy-from-build-lib mechanism picks it up from there.
        return []


setup(
    ext_modules=[
        # Dummy Extension — no sources. CMakeBuildExt.build_extension() does
        # the real work; we just need a hook so build_ext is invoked.
        Extension("opencl_tf.opencl_tf_ops", sources=[]),
    ],
    cmdclass={"build_ext": CMakeBuildExt},
)
