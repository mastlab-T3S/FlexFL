import glob
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WidarDataset(Dataset):
    def __init__(self, data):
        self.data = data

        self.targets = torch.tensor([d[1] for d in self.data], dtype=torch.long)

        self.classes = list(range(22))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        return self.data[idx][0].reshape(22, 20, 20), self.targets[idx]


if __name__ == '__main__':
    # data = torch.load('../data/widar/widar.pkl')
    #
    # t = 1

    files = glob.glob(f'..\\data\\widar\\*.pkl')
    data = []

    for file in files:
        try:
            with open(file, 'rb') as f:
                print(file)
                data.append(torch.load(f))
        except pickle.UnpicklingError as e:
            print(f'Error loading {f}')
        f.close()
    x = [d[0] for d in data]
    x = np.concatenate(x, axis=0, dtype=np.float32)
    x = (x - .0025) / .0119
    y = np.concatenate([d[1] for d in data])
    x = np.transpose(x, (0, 2, 1))
    data = [(x[i], y[i]) for i in range(len(x))]

    dataset = WidarDataset(data)
    t = dataset[0]
