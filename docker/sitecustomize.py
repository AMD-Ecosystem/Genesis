"""Keep CoACD's private C++ symbols isolated from ROCm libraries."""

import ctypes
import os

_original_cdll = ctypes.CDLL


def _isolated_cdll(name, *args, **kwargs):
    if os.path.basename(str(name)).startswith("lib_coacd"):
        kwargs["mode"] = os.RTLD_LOCAL | os.RTLD_DEEPBIND
    return _original_cdll(name, *args, **kwargs)


ctypes.CDLL = _isolated_cdll
