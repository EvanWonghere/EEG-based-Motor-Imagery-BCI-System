"""NumPy compatibility shim for MNE GDF reading on NumPy 2.x.

Import this module before any ``mne.io.read_raw_gdf`` call.
"""

import numpy as np

_PATCHED = False


def patch_numpy_fromstring() -> None:
    """Monkey-patch ``np.fromstring`` so MNE GDF/EDF readers work on NumPy 2."""
    global _PATCHED
    if _PATCHED or not hasattr(np, "fromstring"):
        return
    _PATCHED = True

    _original = np.fromstring

    def _fromstring_compat(s, dtype=float, count=-1, sep=""):
        if sep == "":
            try:
                return np.frombuffer(s, dtype=dtype, count=count)
            except (TypeError, ValueError):
                if isinstance(s, str):
                    return np.frombuffer(
                        s.encode("latin-1"), dtype=dtype, count=count
                    )
                raise
        return _original(s, dtype=dtype, count=count, sep=sep)

    np.fromstring = _fromstring_compat
