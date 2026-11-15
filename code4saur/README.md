Official PyTorch implementation of **Not All Errors Are Equal: Structure-Aware Update Routing for Online Time Series Forecasting**.

**Haihua Xu, Jianpeng Zhao, and Pengyang Wang**


## Structure-Aware Update Routing

SAUR is a backbone-agnostic controller for online multivariate time series forecasting. It uses the temporal structure of delayed prediction residuals to allocate update influence across forecast horizons and output channels.

- **Horizon-aware temporal aggregation:** a shared recency weighting consolidates residual feedback and forms the replay loss.
- **Residual-structure diagnosis:** fast and slow residual statistics track relative error severity, directional consistency, and persistent magnitude escalation.
- **Channel-wise update routing:** positive, mean-normalized channel weights redistribute the replay gradient without adding trainable controller parameters.

SAUR follows a forecast-then-update protocol. Deployment predictions are cached when issued. After a complete target window becomes available, the cached prediction supplies the diagnostic residual, while the current backbone replays the historical input to compute the update loss.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

The implementation uses PyTorch, NumPy, pandas, scikit-learn, and tqdm.

## Data Preparation

Place the forecasting datasets in `./dataset/`, or specify another directory with `--root_path`. Each file should be a CSV containing a `date` column followed by numeric variables, ordered chronologically.

| Dataset | File |
| --- | --- |
| ETTh1 | `ETTh1.csv` |
| ETTh2 | `ETTh2.csv` |
| ETTm1 | `ETTm1.csv` |
| ETTm2 | `ETTm2.csv` |
| Weather | `weather.csv` |
| Exchange | `exchange_rate.csv` |
| Traffic | `traffic.csv` |
| Solar | `solar_AL_dates.csv` |

Dataset metadata is defined in `settings.py`. Data is split chronologically into 20\% for pretraining, 5\% for validation, and 75\% for online evaluation. Standardization is fitted on the training split.

## Usage

The supported forecasting backbones are **DLinear**, **PatchTST**, **iTransformer**, and **TCN**.

Train a backbone and perform online forecasting with SAUR:

```bash
python run.py \
  --model DLinear \
  --dataset ETTh1 \
  --root_path ./dataset \
  --online_method SAUR \
  --seq_len 96 \
  --pred_len 24
```

Run SAUR from a pretrained checkpoint:

```bash
python run.py \
  --model DLinear \
  --dataset ETTh1 \
  --root_path ./dataset \
  --online_method SAUR \
  --seq_len 96 \
  --pred_len 24 \
  --only_test \
  --load_path ./checkpoints/ETTh1_96_24_DLinear_seed2021/checkpoint.pth
```

Set `--use_gpu false` to run on CPU. Use `--model` and `--dataset` to select a backbone and dataset. The controller options use the `--saur_` prefix.

## Code Structure

```text
.
├── data_provider/
├── exp/
│   ├── exp_basic.py
│   ├── exp_main.py
│   ├── exp_online.py
│   └── exp_saur.py
├── layers/
├── models/
├── util/
├── requirements.txt
├── run.py
└── settings.py
```
## Citation

If you use SAUR in your research, please cite:

```bibtex
@inproceedings{xu2027saur,
  title={Not All Errors Are Equal: Structure-Aware Update Routing for Online Time Series Forecasting},
  author={Xu, Haihua and Zhao, Jianpeng and Wang, Pengyang},
  booktitle={Proceedings of the 33rd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2027},
  url={https://openreview.net/forum?id=nC5HjEyMag}
}
```

