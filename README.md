# C3S-Guard Artifact

This folder is the submission-oriented artifact bundle for the C3S-Guard project. It is organized to support inspection, partial reproduction, and figure regeneration for the paper.

## Scope

This artifact is a curated layer on top of the full repository. It contains:

- the core C3S-Guard source snapshot;
- representative attack entry points;
- reproduction scripts for the main CIFAR-10 setting and smoke tests on MNIST/FEMNIST;
- plotting scripts for paper figures;
- sample logs/results used for diagnosis and figure generation.

The artifact is designed to be distributed together with the full repository, or kept inside the repository root as it is now. The scripts in `scripts/` automatically resolve the repository root as the parent of this artifact directory.

The artifact also includes `LICENSE` so that the bundled files can be redistributed with the repository license notice intact.

## Directory Layout

```text
C3S-Guard-artifact/
├── README.md
├── requirements.txt
├── configs/
├── src/
├── attacks/
├── scripts/
├── results/
└── plotting/
```

## What Each Directory Contains

- `configs/`: curated experiment commands and selected YAML attack configs.
- `src/`: snapshot of the C3S-Guard implementation and experiment runner entry points.
- `attacks/`: representative attack entry points used in the paper-facing experiments.
- `scripts/`: one-command launch scripts for reproduction and figure generation.
- `results/`: bundled expected paper figures and representative sample logs/results.
- `plotting/`: plotting utilities for both expected figures and empirical diagnostics.

## Environment

Recommended environment:

- Python `3.10` or `3.11`
- PyTorch `2.0+`
- CUDA-capable GPU for the main CIFAR-10 experiments

Install dependencies from the repository root:

```bash
pip install -r C3S-Guard-artifact/requirements.txt
```

## Main Reproduction Entry Points

From the repository root:

1. Main CIFAR-10 diagnostic run with DCBD:

```bash
bash C3S-Guard-artifact/scripts/run_cifar10_dcbd_main.sh
```

2. CIFAR-10 DCBD + FCT confirmation run:

```bash
bash C3S-Guard-artifact/scripts/run_cifar10_dcbd_fct_main.sh
```

3. MNIST smoke test:

```bash
bash C3S-Guard-artifact/scripts/run_mnist_smoke.sh
```

4. FEMNIST smoke test with EMNIST fallback:

```bash
bash C3S-Guard-artifact/scripts/run_femnist_smoke.sh
```

For Windows users, a PowerShell wrapper is included for the main CIFAR-10 DCBD run:

```powershell
.\C3S-Guard-artifact\scripts\run_cifar10_dcbd_main.ps1
```

## Figure Reproduction

Expected paper figures:

```bash
bash C3S-Guard-artifact/scripts/plot_expected_figures.sh
```

Empirical CTS/TSC and rho_TCD diagnostics:

```bash
bash C3S-Guard-artifact/scripts/plot_empirical_figures.sh
```

## Bundled Sample Results

The `results/` directory includes:

- expected paper figures in PNG/PDF format;
- representative CIFAR-10 JSON/JSONL outputs for C3S-Guard;
- one alpha `0.1` sample log for cross-alpha plotting/debugging;
- small MNIST/FEMNIST smoke outputs when available in the local repository.

These are not claimed to be the full raw experimental archive. They are the minimum bundled materials needed for inspection and for exercising the plotting code paths.

## Notes on Stability

- The most stable detection-only reproduction target in this repository state is the DCBD diagnostic setting.
- The DCBD + FCT setting is included as the main confirmation-layer reproduction target.
- Full repair/execution-chain settings are more sensitive to hyperparameters and safe-model state. They are intentionally not the default artifact entry point.

## Pointers

- Core implementation snapshot: [src](</C:/Users/Zhangzhijun/Desktop/BackdoorBench-main - 副本/C3S-Guard-artifact/src>)
- Reproduction scripts: [scripts](</C:/Users/Zhangzhijun/Desktop/BackdoorBench-main - 副本/C3S-Guard-artifact/scripts>)
- Sample outputs: [results](</C:/Users/Zhangzhijun/Desktop/BackdoorBench-main - 副本/C3S-Guard-artifact/results>)
- Plotting utilities: [plotting](</C:/Users/Zhangzhijun/Desktop/BackdoorBench-main - 副本/C3S-Guard-artifact/plotting>)
