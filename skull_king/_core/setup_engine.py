"""Build the skull_king_engine C extension.

Run from the skull_king/_core/ directory:
    python setup_engine.py build_ext --inplace

Or from the project root:
    python skull_king/_core/setup_engine.py build_ext --inplace

Windows MSVC (must be inside vcvars64.bat env + DISTUTILS_USE_SDK=1):
    $vcvars = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools\\VC\\Auxiliary\\Build\\vcvars64.bat"
    $buildDir = "<project>\\skull_king\\_core"
    cmd /c "`"$vcvars`" && cd /d `"$buildDir`" && set DISTUTILS_USE_SDK=1 && set MSSdk=1 && python setup_engine.py build_ext --inplace"
"""
import sys
import numpy as np
from setuptools import setup, Extension

if sys.platform == "win32":
    # /arch:AVX2 enables AVX2 + FMA via MSVC; /O2 for speed.
    extra_compile_args = ["/O2", "/arch:AVX2", "/W3"]
    extra_link_args    = []
else:
    extra_compile_args = ["-O3", "-march=native", "-Wall", "-ffast-math"]
    extra_link_args    = []

ext = Extension(
    name="skull_king_engine",
    sources=["skull_king_engine.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="skull_king_engine",
    version="1.0",
    description="Full C game engine + MLP + CFR traversal for Skull King",
    ext_modules=[ext],
)
