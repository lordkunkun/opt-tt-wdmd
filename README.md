# OPT-TT-WDMD

Reference code for **Optimized Tensor-Train Windowed Dynamic Mode Decomposition
(OPT-TT-WDMD)**.

This repository accompanies the manuscript:

> **An Algebraically Optimized Tensor-Train Windowed Dynamic Mode Decomposition
> Method for Large-Scale Non-Stationary Flows**

Open-access preprint:
[SSRN paper page](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6922885)

The repository is intentionally small. It exposes the algorithmic ingredients
needed to understand and test the method, while keeping large DNS datasets,
generated result tensors, manuscript figures, and data preparation utilities
outside the public code package.

## Overview

Windowed Dynamic Mode Decomposition (WDMD) is useful for analyzing
non-stationary flow data, but repeatedly applying DMD over many windows becomes
expensive for large tensorized snapshot sequences.

This package implements three closely related code paths:

- dense WDMD on flattened snapshots,
- direct TT-WDMD on a window-local tensor-train representation,
- OPT-TT-WDMD using one global tensor-train representation and temporal-core
  slicing.

The optimized path follows the algebraic organization described in the
manuscript: build a single global tensor-train representation of the full
space-time snapshot tensor, reuse the shared spatial TT cores, slice the final
temporal core for each window, and form each reduced DMD operator from compact
temporal-core factors.

## Repository Structure

```text
.
+-- src/optttwdmd/
|   +-- tensor_train.py      # TT-SVD and TT contraction helpers
|   +-- dmd.py               # Dense DMD reduced-operator utilities
|   +-- windowed.py          # WDMD, direct TT-WDMD, and OPT-TT-WDMD routines
+-- examples/
|   +-- synthetic_demo.py    # Data-free consistency demonstration
+-- docs/
|   +-- data_policy.md
|   +-- manuscript_alignment.md
+-- pyproject.toml
+-- requirements.txt
+-- LICENSE
+-- README.md
```

## Environment

The package requires Python 3.9 or newer and uses PyTorch tensors for the main
linear algebra operations.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For a manual dependency installation:

```bash
pip install -r requirements.txt
```

GPU acceleration is optional. For small demos and most code inspection tasks,
CPU execution is sufficient.

## Quick Start

Run the synthetic consistency demo:

```bash
python examples/synthetic_demo.py
```

The demo constructs a small four-dimensional tensor, compares dense WDMD,
direct TT-WDMD, and OPT-TT-WDMD on the same window, prints the eigenvalue
agreement, and writes a diagnostic plot under `outputs/`.

## Data Shape Convention

The code expects tensorized flow snapshots in the shape:

```text
(n1, n2, ..., nd, nt)
```

The final axis is time. Dense DMD views each snapshot as a flattened vector of
size `n1 * n2 * ... * nd`.

## Manuscript Alignment

This package is organized around the main algorithmic path in the manuscript:

- Sections 2 and 3: dense DMD, TT-SVD, and direct TT-WDMD utilities.
- Section 4: OPT-TT-WDMD with one global TT representation and temporal-core
  slicing.
- Section 5: consistency checks between dense WDMD, direct TT-WDMD, and
  OPT-TT-WDMD.
- Section 6: the boundary-layer application follows the same strict
  OPT-TT-WDMD path, rather than a separate local production variant.

The Section 6 application therefore uses the same method principle as Section
4: construct a global tensor-train representation of the target space-time
flow tensor, reuse the spatial TT factors, slice the temporal core over the
analysis windows, and form the reduced DMD operators from those compact
temporal-core window factors.

## Reproducing Paper Results

The current public package is a method-facing reference implementation. It is
designed to reproduce the algorithmic consistency checks without requiring the
full DNS dataset.

The full runtime benchmark and Section 6 figures require external DNS data and
experiment-specific driver scripts. Any benchmark or application driver should
call the same OPT-TT-WDMD implementation exposed by this package and match the
final manuscript settings, including the training length, window locations, and
frequency normalization.

## Data Availability

The full DNS datasets used in the manuscript are not included because of their
storage size and redistribution constraints. Users who want to run the method
on their own data should provide tensors in the shape described above.

The repository excludes by default:

```text
*.npy
*.npz
*.pt
*.pth
outputs/
figures/
results/
```

See `docs/data_policy.md` for the release boundary.

## License

This repository is released under the MIT License. See `LICENSE`.

## Citation

If you use this code, please cite the open-access SSRN preprint linked above.
A complete journal citation will be added after formal publication.

## Release Notes

- The repository is research-oriented and may evolve with the manuscript.
- Large raw datasets, generated cache files, and manuscript figures are not
  tracked by Git.
