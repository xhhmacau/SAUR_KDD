from data_provider.data_loader import Dataset_Custom, Dataset_Recent
from torch.utils.data import DataLoader
import torch

data_dict = {'custom': Dataset_Custom}
flag2num = {'train': 0, 'val': 1, 'test': 2}

def get_dataset(args, flag, device='cpu', wrap_class=None, borders=None, **kwargs):
    if not hasattr(args, 'timeenc'):
        args.timeenc = 0
    if wrap_class is not None:
        if not isinstance(wrap_class, list):
            wrap_class = [wrap_class]
    border = None
    if borders is not None:
        border = (borders[0][flag2num[flag]], borders[1][flag2num[flag]])
        if flag != 'train':
            if Dataset_Recent in wrap_class:
                start = border[0] - args.pred_len
                border = (max(0, start), border[1])
    data_set = data_dict[args.data](
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=args.timeenc,
        freq=args.freq,
        train_only=args.train_only,
        border=border,
        borders=args.borders if hasattr(args, 'borders') else None,
        ratio=args.ratio if hasattr(args, 'ratio') else None
    )
    if args.pin_gpu and hasattr(data_set, 'data_x'):
        data_set.data_x = torch.as_tensor(data_set.data_x, dtype=torch.float32, device=device)
        data_set.data_y = torch.as_tensor(data_set.data_y, dtype=torch.float32, device=device)
        data_set.data_stamp = torch.as_tensor(data_set.data_stamp, dtype=torch.float32, device=device)
    if wrap_class is not None:
        for cls in wrap_class:
            data_set = cls(data_set, **kwargs)
    print(flag, len(data_set))
    return data_set

def get_dataloader(data_set, args, flag):
    if flag == 'online':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
    elif flag in ['test', 'val']:
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag,
                             num_workers=args.num_workers, drop_last=drop_last, pin_memory=False)
    return data_loader

def data_provider(args, flag, device='cpu', wrap_class=None, **kwargs):
    data_set = get_dataset(args, flag, device, wrap_class=wrap_class, **kwargs)
    data_loader = get_dataloader(data_set, args, flag)
    return data_set, data_loader
