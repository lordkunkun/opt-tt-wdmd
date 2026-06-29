# Manuscript Alignment

This document records how the public code package maps to the manuscript:

> An Algebraically Optimized Tensor-Train Windowed Dynamic Mode Decomposition
> Method for Large-Scale Non-Stationary Flows

Open-access preprint:
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6922885

The repository is a curated method package.

## Sections 2 and 3: DMD and TT-DMD Background

Relevant files:

```text
src/optttwdmd/dmd.py
src/optttwdmd/tensor_train.py
```

These files provide the dense DMD reduced-operator construction, TT-SVD
decomposition, dense reconstruction from TT cores, and compact TT core-chain
contractions used by the direct TT-DMD derivations.

## Section 4: Algebraically Optimized TT-WDMD

Relevant file:

```text
src/optttwdmd/windowed.py
```

The function `opt_tt_wdmd` follows the Section 4 algorithmic path:

1. Build one global TT representation of the full space-time snapshot tensor.
2. Extract the final temporal core.
3. Slice the temporal core for each analysis window.
4. Form shifted compact matrices from the sliced temporal core.
5. Construct the reduced DMD operator from those compact factors.

This is the method path used by benchmark and application drivers associated
with the paper.

## Section 5: Consistency and Runtime Experiments

Relevant files:

```text
src/optttwdmd/windowed.py
examples/synthetic_demo.py
```

The synthetic demo checks the method-level consistency between:

- dense WDMD,
- direct TT-WDMD,
- OPT-TT-WDMD.

The full runtime benchmark requires external DNS tensors and manuscript-specific
experiment settings. Those data files are not redistributed in this repository.

## Section 6: Boundary-Layer Application

The Section 6 application follows the same strict OPT-TT-WDMD path described in
Section 4.

For a public release, any Section 6 driver should call the same optimized
workflow exposed by `optttwdmd.opt_tt_wdmd` or a thin wrapper around it:

```text
global TT decomposition
temporal-core window slicing
compact reduced DMD operator construction
```

The DNS tensors, generated result tensors, and manuscript figures are excluded
from the public repository. See `docs/data_policy.md`.
