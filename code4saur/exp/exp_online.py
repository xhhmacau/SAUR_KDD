from collections import deque

import torch
from torch import optim
from tqdm import tqdm

from data_provider.data_factory import get_dataset, get_dataloader
from exp.exp_main import Exp_Main
from util.metrics import update_metrics, calculate_metrics


class Exp_Online(Exp_Main):

    def _select_online_optimizer(self):
        parameters = (p for p in self.model.parameters() if p.requires_grad)
        learning_rate = self.args.online_learning_rate
        if learning_rate is None:
            learning_rate = self.args.learning_rate
        self.online_optim = getattr(optim, self.args.optim)(parameters, lr=learning_rate)
        return self.online_optim

    def _reset_online_state(self):
        pass

    def _update_online(self, batch, criterion, optimizer, scaler=None):
        optimizer.zero_grad(set_to_none=True)
        outputs = self.forward(batch)
        loss = self.train_loss(criterion, batch, outputs)
        loss.backward()
        optimizer.step()
        return loss.item(), outputs

    def _replay(self, batch, stored_prediction, optimizer):


        loss, _ = self._update_online(batch, self._select_criterion(), optimizer)
        return loss

    def online(self, online_data=None, target_variate=None, phase='test', show_progress=False):
        if self.args.pred_len < 1:
            raise ValueError("pred_len must be positive")
        self.phase = phase
        if online_data is None:
            online_data = get_dataset(
                self.args, phase, self.device,
                wrap_class=self.args.wrap_data_class,
            )
        online_loader = get_dataloader(online_data, self.args, flag='online')
        statistics = {k: 0 for k in ['total', 'y_sum', 'MSE', 'MAE']}
        optimizer = self._select_online_optimizer()
        self._reset_online_state()
        pending = deque()
        if phase == 'test' or show_progress:
            online_loader = tqdm(online_loader, mininterval=60)

        for batch in online_loader:
            self.model.eval()
            with torch.no_grad():
                prediction = self.forward(batch)
                if isinstance(prediction, (tuple, list)):
                    prediction = prediction[0]


                update_metrics(
                    prediction, batch[self.label_position].to(self.device),
                    statistics, target_variate,
                )
            stored_batch = tuple(
                item.detach().clone() if isinstance(item, torch.Tensor) else item
                for item in batch
            )
            pending.append((stored_batch, prediction.detach().clone()))
            if len(pending) > self.args.pred_len:
                delayed_batch, stored_prediction = pending.popleft()
                self.model.train()
                self._replay(delayed_batch, stored_prediction, optimizer)


        if statistics['total'] == 0:
            raise ValueError("online_data contains no forecast samples")
        metrics = calculate_metrics(statistics)
        mse, mae = metrics['MSE'], metrics['MAE']
        if phase == 'test':
            print(f'mse:{mse}, mae:{mae}')
        return mse, mae

    def update_valid(self, valid_data=None):
        if valid_data is None:
            valid_data = get_dataset(
                self.args, 'val', self.device,
                wrap_class=self.args.wrap_data_class,
            )
        self.online(online_data=valid_data, phase='val')
        return []
