# OPT-TT-WDMD

Reference code for the optimized tensor-train windowed DMD method from our paper
*An Algebraically Optimized Tensor-Train Windowed Dynamic Mode Decomposition
Method for Large-Scale Non-Stationary Flows*.

Windowed DMD is a standard tool for non-stationary flows, but running DMD over
many overlapping windows of a large tensorized snapshot sequence gets expensive.
The optimized variant here builds one global tensor-train of the full space-time
tensor, keeps the spatial cores fixed, and only slices the temporal core per
window, so each window solve reduces to a small problem on the temporal-core
factors.

The package exposes three code paths so you can compare them:

- `wdmd` — dense WDMD on flattened snapshots (baseline)
- `direct_tt_wdmd` — TT-WDMD with a separate TT decomposition per window
- `opt_tt_wdmd` — the optimized global-TT version

## Install

Needs Python 3.9+ and PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run the demo

```bash
python examples/synthetic_demo.py
```

It builds a small 4D tensor with a few coherent modes, runs all three methods on
one window, and checks that the eigenvalues agree — they match to about 1e-15. A
spectrum plot is written to `outputs/`.

## Data layout

Snapshots are expected as a tensor of shape `(n1, n2, ..., nd, nt)`, with time on
the last axis. Dense DMD flattens the spatial modes into a vector of length
`n1 * ... * nd`.

## Data and full results

The DNS datasets and the boundary-layer results from the paper are too large to
host here and can't be redistributed, so they are not included; `*.npy`/`*.pt`
files, figures and `outputs/` are gitignored. You only need the demo above to
check the algorithmic consistency. To run the method on your own data, pass a
tensor in the shape above to `opt_tt_wdmd`.

## License

MIT — see `LICENSE`. A full citation will be added once the paper is published;
until then please cite the paper title above.
