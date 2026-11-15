import importlib
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_provider.data_factory import get_dataset
from exp.exp_basic import Exp_Basic
from util.tools import EarlyStopping, adjust_learning_rate
from util.metrics import update_metrics, calculate_metrics
import warnings
warnings.filterwarnings('ignore')

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)

    @property
    def _model(self):
        return self.model

    def _build_model(self, model=None, framework_class=None):
        if model is None:
            model = importlib.import_module(f'models.{self.args.model}').Model(self.args).float()
        if hasattr(self.args, 'load_path') and self.args.load_path:
            self.model_optim = self._select_optimizer(model=model.to(self.device))
            print('Load checkpoints from', self.args.load_path)
            model = self.load_checkpoint(self.args.load_path, model)
            if self.model_optim:
                print('Learning rate of model_optim is', self.model_optim.param_groups[0]['lr'])
        return model

    def _process_batch(self, batch):
        batch_x = batch[0]

        if self.args.model in ['S_Mamba', 'iTransformer']:
            batch_x_mark = batch[2] if len(batch) > 2 else None
            batch_y_mark = batch[3] if len(batch) > 3 else None
            inp = [batch_x, batch_x_mark, None, batch_y_mark]
        else:
            inp = [batch_x]
        return inp

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(vali_loader):
                outputs = self.forward(batch)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                true = batch[self.label_position]
                if not self.args.pin_gpu:
                    true = true.to(self.device)
                loss = criterion(outputs, true)
                total_loss.append(loss.item())
        return np.average(total_loss)

    def train(self, setting, train_data=None, train_loader=None, vali_data=None, vali_loader=None):
        if train_data is None:
            train_data, train_loader = self._get_data(flag='train')
        if vali_data is None and not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
        if self.args.checkpoints:
            path = os.path.join(self.args.checkpoints, setting)
        else:
            path = None
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            for i, batch in enumerate(train_loader):
                model_optim.zero_grad()
                outputs = self.forward(batch)
                loss = self.train_loss(criterion, batch, outputs)
                train_loss.append(loss.item())
                loss.backward()
                model_optim.step()
            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss = np.average(train_loss)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.7f}, Vali Loss: {vali_loss:.7f}")
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            adjust_learning_rate(model_optim, None, epoch + 1, self.args)
        if early_stopping.best_checkpoint:
            self.model.load_state_dict(early_stopping.best_checkpoint)
        if path:
            os.makedirs(path, exist_ok=True)
            torch.save({'model': self.model.state_dict(), 'optimizer': model_optim.state_dict()},
                       os.path.join(path, 'checkpoint.pth'))
        return self.model

    def test(self, setting=None, test_data=None, test_loader=None):
        if test_data is None:
            test_data, test_loader = self._get_data(flag='test')
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                outputs = self.forward(batch)
                update_metrics(outputs, batch[self.label_position].to(self.device), statistics, None)
        metrics = calculate_metrics(statistics)
        print(f"mse:{metrics['MSE']}, mae:{metrics['MAE']}")
        return metrics
