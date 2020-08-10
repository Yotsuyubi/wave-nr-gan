from model import GAN
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    gpus = 1 if torch.cuda.is_available() else 0

    model = GAN(device='cpu')
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=50000,
        # auto_lr_find=True
    )
    trainer.fit(model)

    # noise = torch.randn([1, 64])*torch.randn([1, 1])
    # noise = 0
    # t = np.arange(0, 64, 1)*0.01
    # signal = torch.randn([1, 1])+torch.randn([1, 1])*np.sin(2*np.pi*5*t)+torch.randn([1, 1])*np.sin(2*np.pi*15*t)
    # signal = torch.tensor(signal).reshape([1, -1])
    # plt.plot((noise + signal)[0])
    # plt.show()
