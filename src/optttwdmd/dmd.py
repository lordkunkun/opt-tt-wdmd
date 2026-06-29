"""Dense DMD utilities shared by the windowed routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class DMDResult:
    """Reduced DMD result for one window."""

    eigenvalues: torch.Tensor
    reduced_operator: torch.Tensor
    rank: int
    window_start: int
    window_size: int


def effective_rank(
    singular_values: torch.Tensor,
    rel_tol: float = 1e-10,
    min_rank: int = 1,
    max_rank: Optional[int] = None,
) -> int:
    """Choose a stable reduced rank from singular values."""

    if rel_tol < 0:
        raise ValueError("rel_tol must be non-negative.")
    if min_rank < 0:
        raise ValueError("min_rank must be non-negative.")
    if max_rank is not None and max_rank < 1:
        raise ValueError("max_rank must be positive when provided.")
    if singular_values.numel() == 0:
        return 0
    if singular_values[0] <= 0:
        return 0

    if rel_tol > 0:
        rank = int((singular_values > rel_tol * singular_values[0]).sum().item())
    else:
        rank = singular_values.numel()
    if rank > 0:
        rank = max(rank, min_rank)
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return min(rank, singular_values.numel())


def dmd_from_shifted(
    x: torch.Tensor,
    y: torch.Tensor,
    rel_tol: float = 1e-10,
    max_rank: Optional[int] = None,
    window_start: int = 0,
    window_size: Optional[int] = None,
) -> DMDResult:
    """Compute projected DMD eigenvalues from shifted matrices ``x`` and ``y``.

    ``x`` and ``y`` must have shape ``(state_dimension, snapshots_minus_one)``.
    """

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("DMD inputs must be two-dimensional matrices.")
    if x.shape != y.shape:
        raise ValueError("DMD shifted matrices must have identical shapes.")

    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    rank = effective_rank(s, rel_tol=rel_tol, max_rank=max_rank)
    if rank < 1:
        raise ValueError("DMD input matrix has zero numerical rank.")

    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]

    y_v = y @ vh_r.conj().T
    a_tilde = u_r.conj().T @ y_v @ torch.diag(1.0 / s_r)
    eigenvalues, _ = torch.linalg.eig(a_tilde)

    return DMDResult(
        eigenvalues=eigenvalues,
        reduced_operator=a_tilde,
        rank=rank,
        window_start=window_start,
        window_size=int(window_size or x.shape[1] + 1),
    )


def dmd_from_snapshots(
    snapshots: torch.Tensor,
    rel_tol: float = 1e-10,
    max_rank: Optional[int] = None,
    window_start: int = 0,
) -> DMDResult:
    """Compute DMD from a snapshot matrix with shape ``(state, time)``."""

    if snapshots.ndim != 2:
        raise ValueError("Snapshot data must be a two-dimensional matrix.")
    if snapshots.shape[1] < 2:
        raise ValueError("At least two snapshots are required.")

    x = snapshots[:, :-1]
    y = snapshots[:, 1:]
    return dmd_from_shifted(
        x,
        y,
        rel_tol=rel_tol,
        max_rank=max_rank,
        window_start=window_start,
        window_size=snapshots.shape[1],
    )
