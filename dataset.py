import torch
from torch.utils.data import Dataset
import numpy as np


class SignalWithNoise(Dataset):

    def __init__(self, sample_size, length=512):
        super().__init__()
        self.length = length
        self.sample_size = sample_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        noise = torch.randn([1, self.sample_size])*torch.randn([1, 1])
        t = np.arange(0, self.sample_size, 1)*0.01
        # A + b1*sin(f) + b2*sin(3f)
        signal = torch.randn([1, 1])+torch.randn([1, 1])*np.sin(2*np.pi*5*t)+torch.randn([1, 1])*np.sin(2*np.pi*15*t)
        signal = signal.reshape([1, -1])
        return noise + signal
