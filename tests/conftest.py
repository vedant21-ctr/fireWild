"""
conftest.py - Load PyTorch before anything imports LightGBM or XGBoost.

Those libraries bring in Homebrew's OpenMP runtime. On macOS, loading it before PyTorch's own copy makes PyTorch
tensor operations segfault, which used to crash the suite when tests from both families ran in one process.
"""

import torch  # noqa: F401
