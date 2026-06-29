"""Synthetic OPT-TT-WDMD consistency demo.

This demo does not use any external DNS data. It builds a small tensorized
snapshot sequence with a few coherent temporal components, then compares:

1. dense windowed DMD,
2. direct TT-WDMD, and
3. optimized TT-WDMD based on global temporal-core slicing.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from optttwdmd import direct_tt_wdmd, opt_tt_wdmd, wdmd


def make_synthetic_tensor(
    spatial_shape: tuple[int, int, int] = (8, 7, 5),
    nt: int = 140,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Construct a small tensorized dataset with coherent oscillations."""

    generator = torch.Generator().manual_seed(13)
    spatial_modes = [
        torch.randn(*spatial_shape, dtype=dtype, generator=generator)
        for _ in range(5)
    ]

    t = torch.arange(nt, dtype=dtype)
    temporal = [
        torch.cos(0.18 * t),
        torch.sin(0.18 * t),
        (0.995**t) * torch.cos(0.07 * t),
        (0.995**t) * torch.sin(0.07 * t),
        0.998**t,
    ]

    data = torch.zeros(*spatial_shape, nt, dtype=dtype)
    for mode, coeff in zip(spatial_modes, temporal):
        data = data + mode[..., None] * coeff

    return data


def max_nearest_distance(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Return a symmetric order-independent eigenvalue mismatch."""

    ref = reference.detach().cpu().numpy()
    cand = candidate.detach().cpu().numpy()
    if ref.size == 0 or cand.size == 0:
        return float("nan")

    def directed_distance(values: np.ndarray, targets: np.ndarray) -> float:
        distances = []
        for value in values:
            distances.append(float(np.min(np.abs(targets - value))))
        return max(distances)

    return max(directed_distance(ref, cand), directed_distance(cand, ref))


def plot_spectrum(results, output_path: Path) -> None:
    """Plot eigenvalues for the three methods."""

    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot(np.cos(theta), np.sin(theta), "--", color="0.65", linewidth=1.0)

    styles = [
        ("Dense WDMD", results["dense"].eigenvalues, "k", "+"),
        ("Direct TT-WDMD", results["direct"].eigenvalues, "tab:blue", "o"),
        ("OPT-TT-WDMD", results["optimized"].eigenvalues, "tab:red", "*"),
    ]
    for label, eigs, color, marker in styles:
        values = eigs.detach().cpu().numpy()
        if marker == "o":
            ax.scatter(
                values.real,
                values.imag,
                label=label,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                s=42,
            )
        else:
            ax.scatter(values.real, values.imag, label=label, marker=marker, c=color, s=55)

    ax.axhline(0.0, color="0.25", linewidth=0.7)
    ax.axvline(0.0, color="0.25", linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re(lambda)")
    ax.set_ylabel("Im(lambda)")
    ax.set_title("Synthetic window eigenvalue check")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    tensor = make_synthetic_tensor()
    window_size = 56
    start = 30
    rel_tol = 1e-8

    dense = wdmd(tensor, window_size=window_size, starts=[start], rel_tol=rel_tol)[0]
    direct = direct_tt_wdmd(tensor, window_size=window_size, starts=[start], rel_tol=rel_tol)[0]
    optimized = opt_tt_wdmd(tensor, window_size=window_size, starts=[start], rel_tol=rel_tol)[0]

    results = {
        "dense": dense,
        "direct": direct,
        "optimized": optimized,
    }

    print("Synthetic OPT-TT-WDMD consistency demo")
    print(f"tensor shape: {tuple(tensor.shape)}")
    print(f"window start: {start}")
    print(f"window size:  {window_size}")
    print(f"dense rank:   {dense.rank}")
    print(f"direct rank:  {direct.rank}")
    print(f"opt rank:     {optimized.rank}")
    print(
        "max nearest eigenvalue distance, dense vs direct: "
        f"{max_nearest_distance(dense.eigenvalues, direct.eigenvalues):.3e}"
    )
    print(
        "max nearest eigenvalue distance, dense vs opt:    "
        f"{max_nearest_distance(dense.eigenvalues, optimized.eigenvalues):.3e}"
    )

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "synthetic_eigenvalues.png"
    plot_spectrum(results, output_path)
    print(f"saved plot: {output_path}")


if __name__ == "__main__":
    main()
