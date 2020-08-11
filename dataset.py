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
        t = torch.arange(0, self.sample_size, 1)*0.001
        signal = torch.randn([1, 1])+torch.randn([1, 1])*np.sin(2*np.pi*1*t+torch.randn([1, 1]).numpy())
        signal = signal.reshape([1, -1])
        spike = torch.roll((torch.sin(2*np.pi*4*t)/(2*np.pi*4*t+1E-9))*5, shifts=torch.randint(0, self.sample_size, (1,)).item()).reshape([1, -1])
        if torch.rand([1]).item() < 0.8:
            spike = spike * 0.
        return self.norm(noise + signal + spike)

    def norm(self, x):
        return (x - x.min()) / (x.max() - x.min())
