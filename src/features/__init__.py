"""Feature extraction: CSP, FBCSP, band power, and feature selection."""

from __future__ import annotations

from typing import Any

from src.features.band_power import BandPowerExtractor
from src.features.csp import CSPExtractor
from src.features.fbcsp import FBCSPExtractor

EXTRACTOR_REGISTRY: dict[str, type] = {
    "csp": CSPExtractor,
    "fbcsp": FBCSPExtractor,
    "band_power": BandPowerExtractor,
}


def create_extractor(config: dict[str, Any], sfreq: float = 250.0):
    """Instantiate a feature extractor from a ``features`` config block.

    Parameters
    ----------
    config : dict
        The ``features`` section of an experiment config.  Expected keys::

            method: "csp"   # csp | fbcsp | band_power | null
            params:
              n_components: 4
              ...
    sfreq : float
        Sampling rate, forwarded to extractors that need it.

    Returns
    -------
    extractor or None
        ``None`` if ``method`` is ``null`` (deep learning path).
    """
    method = config.get("method")
    if method is None:
        return None

    if method not in EXTRACTOR_REGISTRY:
        raise ValueError(
            f"Unknown feature method {method!r}. "
            f"Available: {list(EXTRACTOR_REGISTRY.keys())}"
        )

    params = dict(config.get("params") or {})

    # Inject sfreq for extractors that need it
    cls = EXTRACTOR_REGISTRY[method]
    if method == "csp":
        # CSPExtractor accepts: n_components, log, reg
        filtered = {k: params[k] for k in ("n_components", "log", "reg") if k in params}
        return cls(**filtered)

    if method == "fbcsp":
        # Map nested selection config to flat params
        selection = params.pop("selection", None)
        if selection:
            params.setdefault("selection_method", selection.get("method", "mutual_info"))
            params.setdefault("n_select", selection.get("n_select", 8))
        params.setdefault("sfreq", sfreq)
        if "filter_bands" in params:
            params["filter_bands"] = [tuple(b) for b in params["filter_bands"]]
        # FBCSPExtractor accepts: filter_bands, n_components, selection_method, n_select, sfreq
        known = ("filter_bands", "n_components", "selection_method", "n_select", "sfreq")
        filtered = {k: params[k] for k in known if k in params}
        return cls(**filtered)

    if method == "band_power":
        params.setdefault("sfreq", sfreq)
        known = ("bands", "sfreq")
        filtered = {k: params[k] for k in known if k in params}
        return cls(**filtered)

    return cls(**params)


__all__ = [
    "CSPExtractor",
    "FBCSPExtractor",
    "BandPowerExtractor",
    "EXTRACTOR_REGISTRY",
    "create_extractor",
]
