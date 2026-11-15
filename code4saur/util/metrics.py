import numpy as np

def update_metrics(pred, label, statistics, target_variate=None):
    if isinstance(pred, tuple):
        pred = pred[0]
    if target_variate is not None:
        pred = pred[:, :, target_variate]
        if label.dim() == 3:
            label = label[:, :, target_variate]
    balance = pred - label
    statistics['y_sum'] += label.abs().sum().item()
    statistics['total'] += len(label.view(-1))
    statistics['MAE'] += balance.abs().sum().item()
    statistics['MSE'] += (balance ** 2).sum().item()

def calculate_metrics(statistics):
    MSE, MAE, total = statistics['MSE'], statistics['MAE'], statistics['total']
    return {'MSE': MSE / total, 'MAE': MAE / total}

def metric(pred, true):
    mae = np.mean(np.abs(pred - true))
    mse = np.mean((pred - true) ** 2)
    return mae, mse
