"""Build the skull_king_core C extension.

Run from the skull_king/_core/ directory:
    python setup.py build_ext --inplace

Or from the project root:
    python skull_king/_core/setup.py build_ext --inplace

The resulting .pyd (Windows) / .so (Linux) is placed next to setup.py so that
    from skull_king._core.skull_king_core import resolve_trick, legal_cards_mask
works correctly.
"""
import sys
from setuptools import setup, Extension

# Aggressive optimisation flags — safe for pure arithmetic C code.
if sys.platform == "win32":
    extra_compile_args = ["/O2", "/W3"]
    extra_link_args    = []
else:
    extra_compile_args = ["-O3", "-march=native", "-Wall"]
    extra_link_args    = []

ext = Extension(
    # Module lives at skull_king/_core/skull_king_core.{pyd,so}
    # Installed with --inplace so the import path matches.
    name="skull_king_core",
    sources=["skull_king_core.c"],
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
)

setup(
    name="skull_king_core",
    version="1.0",
    description="C-accelerated hot-path game logic for Skull King CFR training",
    ext_modules=[ext],
)
