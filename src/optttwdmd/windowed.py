"""Windowed DMD variants used by the OPT-TT-WDMD reference package."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .dmd import DMDResult, dmd_from_shifted, dmd_from_snapshots, effective_rank
from .tensor_train import tt_contract_spatial, tt_svd


def window_starts(
    total_snapshots: int,
    window_size: int,
    step: int = 1,
    max_windows: Optional[int] = None,
) -> List[int]:
    """Return valid window start indices."""

    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    if step < 1:
        raise ValueError("step must be positive.")
    if max_windows is not None and max_windows < 0:
        raise ValueError("max_windows must be non-negative when provided.")
    if total_snapshots < window_size:
        return []

    starts = list(range(0, total_snapshots - window_size + 1, step))
    if max_windows is not None:
        starts = starts[: int(max_windows)]
    return starts


def _flatten_snapshots(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1, tensor.shape[-1])


def _resolve_starts(
    total_snapshots: int,
    window_size: int,
    starts: Optional[Sequence[int]],
    step: int,
    max_windows: Optional[int],
) -> List[int]:
    if starts is None:
        return window_starts(
            total_snapshots,
            window_size,
            step=step,
            max_windows=max_windows,
        )

    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    if step < 1:
        raise ValueError("step must be positive.")
    if max_windows is not None and max_windows < 0:
        raise ValueError("max_windows must be non-negative when provided.")

    resolved = [int(start) for start in starts]
    for start in resolved:
        if start < 0:
            raise ValueError("Window starts must be non-negative.")
        if start + window_size > total_snapshots:
            raise ValueError("Window start plus window_size exceeds available snapshots.")
    if max_windows is not None:
        resolved = resolved[: int(max_windows)]
    return resolved


def wdmd(
    tensor: torch.Tensor,
    window_size: int,
    starts: Optional[Sequence[int]] = None,
    step: int = 1,
    max_windows: Optional[int] = None,
    rel_tol: float = 1e-10,
) -> List[DMDResult]:
    """Run dense windowed DMD on a tensorized snapshot sequence."""

    if tensor.ndim < 2:
        raise ValueError("Input tensor must contain at least one spatial mode and time.")
    total = tensor.shape[-1]
    starts = _resolve_starts(total, window_size, starts, step, max_windows)

    snapshot_matrix = _flatten_snapshots(tensor)
    results: List[DMDResult] = []
    for start in starts:
        window = snapshot_matrix[:, start : start + window_size]
        results.append(
            dmd_from_snapshots(
                window,
                rel_tol=rel_tol,
                window_start=int(start),
            )
        )
    return results


def direct_tt_wdmd(
    tensor: torch.Tensor,
    window_size: int,
    starts: Optional[Sequence[int]] = None,
    step: int = 1,
    max_windows: Optional[int] = None,
    rel_tol: float = 1e-10,
) -> List[DMDResult]:
    """Run a direct TT-WDMD baseline by decomposing each window separately."""

    if tensor.ndim < 2:
        raise ValueError("Input tensor must contain at least one spatial mode and time.")
    total = tensor.shape[-1]
    starts = _resolve_starts(total, window_size, starts, step, max_windows)

    results: List[DMDResult] = []
    for start in starts:
        window = tensor[..., start : start + window_size]
        x_tt = tt_svd(window[..., :-1], rel_tol=0.0)
        y_tt = tt_svd(window[..., 1:], rel_tol=0.0)

        x_core = x_tt.cores[-1].squeeze(-1)
        y_core = y_tt.cores[-1].squeeze(-1)

        u, s, vh = torch.linalg.svd(x_core, full_matrices=False)
        rank = effective_rank(s, rel_tol=rel_tol)
        if rank < 1:
            raise ValueError("Direct TT-WDMD input matrix has zero numerical rank.")
        u_r = u[:, :rank]
        s_r = s[:rank]
        vh_r = vh[:rank, :]

        spatial_overlap = tt_contract_spatial(x_tt.cores[:-1], y_tt.cores[:-1])
        y_v = torch.tensordot(y_core, vh_r.conj(), dims=([-1], [-1]))
        projected_overlap = torch.tensordot(
            spatial_overlap,
            u_r.conj(),
            dims=([0], [0]),
        ).T
        a_tilde = projected_overlap @ y_v @ torch.diag(1.0 / s_r)
        eigenvalues, _ = torch.linalg.eig(a_tilde)

        results.append(
            DMDResult(
                eigenvalues=eigenvalues,
                reduced_operator=a_tilde,
                rank=rank,
                window_start=int(start),
                window_size=window_size,
            )
        )
    return results


def opt_tt_wdmd(
    tensor: torch.Tensor,
    window_size: int,
    starts: Optional[Sequence[int]] = None,
    step: int = 1,
    max_windows: Optional[int] = None,
    rel_tol: float = 1e-10,
) -> List[DMDResult]:
    """Run OPT-TT-WDMD using one global TT and temporal-core slicing."""

    if tensor.ndim < 2:
        raise ValueError("Input tensor must contain at least one spatial mode and time.")
    total = tensor.shape[-1]
    starts = _resolve_starts(total, window_size, starts, step, max_windows)

    global_tt = tt_svd(tensor, rel_tol=0.0)
    time_core = global_tt.cores[-1].squeeze(-1)

    results: List[DMDResult] = []
    for start in starts:
        core_window = time_core[:, start : start + window_size]
        x_core = core_window[:, :-1]
        y_core = core_window[:, 1:]
        results.append(
            dmd_from_shifted(
                x_core,
                y_core,
                rel_tol=rel_tol,
                window_start=int(start),
                window_size=window_size,
            )
        )
    return results
