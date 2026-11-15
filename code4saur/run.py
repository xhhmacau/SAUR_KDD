import argparse
import csv
import os
import random

import numpy as np
import torch

import settings
from exp import Exp_Main, Exp_Online, Exp_SAUR


def boolean(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise ValueError(value)


parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=["DLinear", "PatchTST", "iTransformer", "TCN"])
parser.add_argument("--dataset", required=True, choices=list(settings.data_settings))
parser.add_argument("--root_path", default="./dataset")
parser.add_argument("--checkpoints", default="./checkpoints")
parser.add_argument("--load_path")
parser.add_argument("--output", default="./results/results.csv")
parser.add_argument("--online_method", choices=["Online", "SAUR"])
parser.add_argument("--only_test", action="store_true")
parser.add_argument("--train_only", action="store_true")
parser.add_argument("--features", default="M", choices=["M", "S", "MS"])
parser.add_argument("--target", default="OT")
parser.add_argument("--freq", default="h")
parser.add_argument("--seq_len", type=int, default=60)
parser.add_argument("--label_len", type=int, default=30)
parser.add_argument("--pred_len", type=int, default=24)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--train_epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--online_learning_rate", type=float, default=0.0001)
parser.add_argument("--optim", default="Adam")
parser.add_argument("--lradj", default="type3")
parser.add_argument("--seed", type=int, default=2021)
parser.add_argument("--use_gpu", type=boolean, default=True)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument("--pin_gpu", type=boolean, default=False)
parser.add_argument("--individual", action="store_true")
parser.add_argument("--e_layers", type=int, default=3)
parser.add_argument("--n_heads", type=int, default=16)
parser.add_argument("--d_model", type=int, default=128)
parser.add_argument("--d_ff", type=int, default=256)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--activation", default="gelu")
parser.add_argument("--factor", type=int, default=1)
parser.add_argument("--use_norm", type=int, default=1)
parser.add_argument("--output_attention", type=int, default=0)
parser.add_argument("--class_strategy", default="projection")
parser.add_argument("--embed", default="timeF")
parser.add_argument("--fc_dropout", type=float, default=0.2)
parser.add_argument("--head_dropout", type=float, default=0.0)
parser.add_argument("--patch_len", type=int, default=16)
parser.add_argument("--stride", type=int, default=8)
parser.add_argument("--padding_patch", default="end")
parser.add_argument("--revin", type=int, default=1)
parser.add_argument("--affine", type=int, default=0)
parser.add_argument("--subtract_last", type=int, default=0)
parser.add_argument("--decomposition", type=int, default=0)
parser.add_argument("--kernel_size", type=int, default=25)
parser.add_argument("--saur_alpha", type=float, default=0.3)
parser.add_argument("--saur_beta", type=float, default=0.05)
parser.add_argument("--saur_gamma", type=float, default=0.2)
parser.add_argument("--saur_kappa", type=float, default=2.0)
parser.add_argument("--saur_a", type=float, default=4.0)
parser.add_argument("--saur_q", type=float, default=0.5)
parser.add_argument("--saur_delta", type=float, default=2.0)
args = parser.parse_args()
args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
args.wrap_data_class = []
args.timeenc = 0
args.data = "custom"
args.data_path = settings.data_settings[args.dataset]["data"]
args.enc_in, args.c_out = settings.data_settings[args.dataset][args.features]
settings.get_borders(args)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
setting = f"{args.dataset}_{args.seq_len}_{args.pred_len}_{args.model}_seed{args.seed}"
if args.load_path is None:
    candidate = os.path.join(args.checkpoints, setting, "checkpoint.pth")
    if os.path.exists(candidate):
        args.load_path = candidate
if args.only_test and args.load_path is None:
    parser.error("--only_test requires --load_path or an existing pretrained checkpoint")
experiment_class = Exp_Main if args.online_method is None else Exp_Online if args.online_method == "Online" else Exp_SAUR
experiment = experiment_class(args)
if not args.only_test:
    experiment.train(setting)
if args.online_method:

    mse, mae = experiment.online()
else:
    metrics = experiment.test(setting)
    mse, mae = metrics["MSE"], metrics["MAE"]
directory = os.path.dirname(args.output)
if directory:
    os.makedirs(directory, exist_ok=True)
exists = os.path.exists(args.output)
with open(args.output, "a", newline="") as handle:
    writer = csv.writer(handle)
    if not exists:
        writer.writerow(["method", "dataset", "model", "seq_len", "pred_len", "seed", "mse", "mae"])
    writer.writerow([args.online_method or "Offline", args.dataset, args.model, args.seq_len, args.pred_len, args.seed, mse, mae])
