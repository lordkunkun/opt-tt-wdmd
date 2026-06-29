# Data Policy

The public repository should not contain raw DNS data, processed DNS tensors,
large PyTorch result files, or generated manuscript figures.

## Excluded By Default

```text
*.npy
*.npz
*.pt
*.pth
*.png
*.pdf
*.gif
outputs/
figures/
results/
```

## Expected External Data Format

External flow data should be supplied as a tensor with shape:

```text
(n1, n2, ..., nd, nt)
```

where the final axis is time. The DNS experiments in the manuscript use
four-dimensional flow tensors, but the implementation accepts any number of
spatial tensor modes.

## Reproducibility Boundary

The public package demonstrates the algorithm and provides the code path used
for windowed DMD comparisons. Dataset redistribution, JHTDB query scripts, and
data preparation utilities are outside the public release boundary.
