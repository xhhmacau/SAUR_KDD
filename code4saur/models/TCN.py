import torch
import torch.nn as nn

from layers.ts2vec.encoder import TS2VecEncoderWrapper, TSEncoder


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        encoder = TSEncoder(input_dims=args.enc_in + 7,
                            output_dims=320,
                            hidden_dims=64,
                            depth=10)
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true')
        self.pred_len = args.pred_len
        self.dim = args.c_out * args.pred_len


        self.regressor = nn.Linear(320, self.dim)

    def forward(self, x, x_mark=None):
        if x_mark is None:
            x_mark = torch.zeros(*x.shape[:2], 7, device=x.device)
        x = torch.cat([x, x_mark], dim=-1)
        rep = self.encoder(x)
        y = self.regressor(rep)
        y = y.reshape(len(y), self.pred_len, -1)
        return y
