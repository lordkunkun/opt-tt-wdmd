"""Reference implementation for optimized tensor-train windowed DMD."""

from .dmd import DMDResult, dmd_from_shifted, dmd_from_snapshots
from .tensor_train import TTDecomposition, tt_contract_spatial, tt_svd, tt_to_dense
from .windowed import direct_tt_wdmd, opt_tt_wdmd, window_starts, wdmd

__all__ = [
    "DMDResult",
    "TTDecomposition",
    "direct_tt_wdmd",
    "dmd_from_shifted",
    "dmd_from_snapshots",
    "opt_tt_wdmd",
    "tt_contract_spatial",
    "tt_svd",
    "tt_to_dense",
    "wdmd",
    "window_starts",
]
