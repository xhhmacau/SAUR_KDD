import os
import torch
import numpy as np
from torch import optim
import torch.nn as nn
from data_provider.data_factory import data_provider

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.label_position = 1
        self.device = self._acquire_device()
        self.wrap_data_kwargs = {}
        self.model_optim = None
        model = self._build_model()
        if model is not None:
            self.model = model.to(self.device)
            self.model_optim = self._select_optimizer()

    def _acquire_device(self):
        if self.args.use_gpu:
            if self.args.pin_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
                device = torch.device('cuda:0')
                print('Use GPU: cuda:{} (visible as cuda:0)'.format(self.args.gpu))
            else:
                device = torch.device('cuda:{}'.format(self.args.gpu))
                print('Use GPU: cuda:{}'.format(self.args.gpu))
            torch.cuda.set_device(device)
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self, model=None, framework_class=None):
        raise NotImplementedError

    def _get_data(self, flag, **kwargs):
        data_set, data_loader = data_provider(args=self.args, flag=flag, device=self.device,
                                              wrap_class=self.args.wrap_data_class, **self.wrap_data_kwargs, **kwargs)
        return data_set, data_loader

    def _select_optimizer(self, filter_frozen=True, return_self=True, model=None):
        if return_self and self.model_optim is not None:
            return self.model_optim
        params = self.model.parameters() if model is None else model.parameters()
        if filter_frozen:
            params = filter(lambda p: p.requires_grad, params)
        model_optim = getattr(optim, self.args.optim)(params, lr=self.args.learning_rate)
        if return_self:
            self.model_optim = model_optim
        return model_optim

    def _select_criterion(self):
        return nn.MSELoss()

    def _process_batch(self, batch):
        return batch

    def forward(self, batch):
        if not self.args.pin_gpu:
            batch = [batch[i].to(self.device) if isinstance(batch[i], torch.Tensor) and i != self.label_position
                     else batch[i] for i in range(len(batch))]
        inp = self._process_batch(batch)
        outputs = self.model(*inp)
        return outputs

    def train_loss(self, criterion, batch, outputs):
        batch_y = batch[1]
        if not self.args.pin_gpu:
            batch_y = batch_y.to(self.device)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return criterion(outputs, batch_y)

    def state_dict(self):
        return {'model': self.model.state_dict(), 'optimizer': self.model_optim.state_dict() if self.model_optim else None}

    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict['model'], strict=strict)
        if self.model_optim and state_dict.get('optimizer'):
            self.model_optim.load_state_dict(state_dict['optimizer'])

    def load_checkpoint(self, path, model=None):
        if model is None:
            model = self.model
        checkpoint = self.load_checkpoint_file(path)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
            if self.model_optim and 'optimizer' in checkpoint:
                self.model_optim.load_state_dict(checkpoint['optimizer'])
        else:
            model.load_state_dict(checkpoint, strict=False)
        return model

    def load_checkpoint_file(self, path):
        return torch.load(path, map_location=self.device)
