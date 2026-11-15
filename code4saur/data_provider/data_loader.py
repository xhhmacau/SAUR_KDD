from typing import Union
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from util.timefeatures import time_features
import warnings
warnings.filterwarnings('ignore')

def get_alldata(filename='weather.csv', root_path='./'):
    path = os.path.join(root_path, filename)
    if filename.endswith('.csv'):
        df = pd.read_csv(path)
    return df

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='S', data_path='weather.csv',
                 ratio=None, borders=None, target='OT', scale=True, timeenc=0, freq='h',
                 train_only=False, border=None):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 96
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.train_only = train_only
        self.border = border
        self.split_ratio = (0.7, 0.2) if ratio is None else ratio
        self.provided_borders = borders
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = get_alldata(self.data_path, self.root_path)
        cols = list(df_raw.columns)
        if self.features == 'S':
            cols.remove(self.target)
        cols.remove('date')

        if self.provided_borders is not None:
            border1s, border2s = self.provided_borders
        else:
            num_train = int(len(df_raw) * (self.split_ratio[0] if not self.train_only else 1))
            num_test = int(len(df_raw) * self.split_ratio[1])
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
        self.borders = (border1s, border2s)
        if self.border is not None:
            border1, border2 = self.border
        else:
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]
        if self.features == 'M' or self.features == 'MS':
            df_raw = df_raw[['date'] + cols]
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_raw = df_raw[['date'] + cols + [self.target]]
            df_data = df_raw[[self.target]]
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, self.timeenc, freq=self.freq)
        data = data.astype(np.float32)
        self.data = data
        self.border = (border1, border2)
        self.data_x = data[border1:border2]
        if self.features == 'MS':
            self.data_y = data[:, -1][border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp.astype(np.float32)
        self.data_x = torch.from_numpy(self.data_x).float()
        self.data_y = torch.from_numpy(self.data_y).float()
        self.data_stamp = torch.from_numpy(self.data_stamp).float()

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_end = s_end + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[s_end:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[s_end:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

class Dataset_Recent(Dataset):
    def __init__(self, dataset, gap: Union[int, tuple, list], recent_num=1, **kwargs):
        super().__init__()
        self.more = gap - recent_num + 1
        self.dataset = dataset
        self.gap = gap
        self.recent_num = recent_num

    def __getitem__(self, index):
        if self.recent_num == 1:
            return self.dataset[index], self.dataset[index + self.gap]
        else:
            current_data = self.dataset[index + self.gap + self.recent_num - 1]
            recent_data = tuple([] for _ in range(len(current_data)))
            for past in range(self.recent_num):
                for j, past_data in enumerate(self.dataset[index + past]):
                    recent_data[j].append(past_data)
            recent_data = tuple(torch.stack(recent_d, 0) for recent_d in recent_data)
            return recent_data, current_data

    def __len__(self):
        return len(self.dataset) - self.more
