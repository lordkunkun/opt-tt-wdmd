"""Small tensor-train helpers used by the OPT-TT-WDMD demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch


@dataclass
class TTDecomposition:
    """Tensor-train cores and interface singular values."""

    cores: List[torch.Tensor]
    singular_values: List[torch.Tensor]

    @property
    def ranks(self) -> List[int]:
        ranks = [int(self.cores[0].shape[0])]
        ranks.extend(int(core.shape[-1]) for core in self.cores)
        return ranks

    @property
    def max_rank(self) -> int:
        return max(self.ranks)


def _rank_from_singular_values(
    singular_values: torch.Tensor,
    max_rank: Optional[int],
    rel_tol: float,
) -> int:
    if singular_values.numel() == 0:
        return 0
    if singular_values[0] <= 0:
        return 1

    rank = singular_values.numel()
    if rel_tol > 0:
        rank = int((singular_values > rel_tol * singular_values[0]).sum().item())
        rank = max(rank, 1)
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    return rank


def tt_svd(
    tensor: torch.Tensor,
    max_rank: Optional[int] = None,
    rel_tol: float = 0.0,
) -> TTDecomposition:
    """Compute a left-orthonormal TT-SVD decomposition.

    Parameters
    ----------
    tensor:
        Input tensor with any order >= 2.
    max_rank:
        Optional maximum TT rank. ``None`` keeps the full numerical rank.
    rel_tol:
        Optional relative singular-value cutoff. The default ``0`` keeps all
        singular values, which is the algebraic-equivalence setting used in the
        manuscript.
    """

    if tensor.ndim < 2:
        raise ValueError("TT-SVD requires a tensor with at least two modes.")
    if rel_tol < 0:
        raise ValueError("rel_tol must be non-negative.")
    if max_rank is not None and max_rank < 1:
        raise ValueError("max_rank must be positive when provided.")

    x = tensor
    dims = tuple(int(v) for v in x.shape)
    left_rank = 1
    cores: List[torch.Tensor] = []
    singular_values: List[torch.Tensor] = []

    for mode_size in dims[:-1]:
        matrix = x.reshape(left_rank * mode_size, -1)
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        rank = _rank_from_singular_values(s, max_rank=max_rank, rel_tol=rel_tol)

        u = u[:, :rank]
        s = s[:rank]
        vh = vh[:rank, :]

        cores.append(u.reshape(left_rank, mode_size, rank).contiguous())
        singular_values.append(s)
        x = torch.diag(s.to(vh.dtype)) @ vh
        left_rank = rank

    cores.append(x.reshape(left_rank, dims[-1], 1).contiguous())
    return TTDecomposition(cores=cores, singular_values=singular_values)


def tt_to_dense(cores: Iterable[torch.Tensor]) -> torch.Tensor:
    """Reconstruct a dense tensor from TT cores."""

    core_list = list(cores)
    if not core_list:
        raise ValueError("At least one TT core is required.")

    result = core_list[0]
    for core in core_list[1:]:
        result = torch.tensordot(result, core, dims=([-1], [0]))

    shape = [int(core.shape[1]) for core in core_list]
    return result.reshape(*shape)


def tt_contract_spatial(
    left_cores: Iterable[torch.Tensor],
    right_cores: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Contract two TT core chains over all physical modes.

    The left and right boundary ranks are retained. For spatial chains with
    left boundary rank one, the result has shape ``(r_left, r_right)`` and is
    the compact overlap matrix used in the direct TT-WDMD reduced operator.
    """

    left = list(left_cores)
    right = list(right_cores)
    if len(left) != len(right):
        raise ValueError("Both TT chains must have the same number of cores.")
    if not left:
        return torch.ones((1, 1))

    device = left[0].device
    dtype = left[0].dtype
    value = torch.ones((left[0].shape[0], right[0].shape[0]), dtype=dtype, device=device)

    for left_core, right_core in zip(left, right):
        if left_core.shape[1] != right_core.shape[1]:
            raise ValueError("Physical mode sizes must match for TT contraction.")
        value = torch.einsum("ac,aib,cid->bd", value, left_core.conj(), right_core)

    return value
