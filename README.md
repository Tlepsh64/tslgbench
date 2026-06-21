# tslgbench

A comparison of gradient estimators for end-to-end sparse graph learning in
time-series forecasting: SNS, Double Control Variates (DoubleCV), IMLE, and
AIMLE, evaluated as drop-in graph samplers within the framework of
"Sparse Graph Learning from Spatiotemporal Time Series" (Cini, Zambon, Alippi;
JMLR 2023).

This repository contains only the core library and experiment entry points —
no logs, run outputs, datasets, or scratch notebooks. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution of adapted
third-party code.

## Directory structure

```
.
├── lib/
│   ├── datasets/             # synthetic dataset generators (GPVAR, etc.)
│   ├── nn/                   # forecasting models and graph samplers
│   │   └── graph_samplers/
│   │       └── imle/         # IMLE / AIMLE gradient estimators
│   ├── gradient_estimators/  # DoubleCV estimator
│   ├── predictors/           # end-to-end predictor wiring a forecaster + sampler
│   └── utils/
├── experiments/
│   ├── config/                # Hydra-style experiment configs
│   ├── run_synthetic.py       # GPVAR experiments
│   └── run_real.py            # AQI / real-data experiments
├── conda_env.yaml
└── default_config.yaml
```

## Requirements

```bash
conda env create -f conda_env.yaml
conda activate sgl
```

This library relies on [tsl (torch-spatiotemporal)](https://torch-spatiotemporal.readthedocs.io/en/latest/).

## Running experiments

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.run_synthetic config=defaults
```

Edit or add a config under `experiments/config/synthetic/` to switch graph
samplers (`sns`, `doublecv`, `imle`, `aimle`) or model settings. Real-data
(PEMS-BAY / METR-LA) experiments are run via `experiments/run_real.py`.

## Acknowledgments

This work builds directly on the codebase and framework released by Cini,
Zambon, and Alippi for:

```
@article{cini2023sparse,
  title={Sparse Graph Learning from Spatiotemporal Time Series},
  author={Cini, Andrea and Zambon, Daniele and Alippi, Cesare},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={242},
  pages={1--36},
  year={2023}
}
```

It also incorporates adapted code from torch-imle, torch-adaptive-imle,
ConcreteDropout, and double-cv — see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.
