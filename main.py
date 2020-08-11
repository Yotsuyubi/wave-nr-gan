from model import GAN
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    gpus = 1 if torch.cuda.is_available() else 0
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GAN(device=dev)
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=50000,
        # auto_lr_find=True
    )
    trainer.fit(model)


    # noise = torch.randn([1, 4096])*torch.randn([1, 1])
    # t = np.arange(0, 4096, 1)*0.001
    # signal = torch.randn([1, 1])+torch.randn([1, 1])*np.sin(2*np.pi*1*t+torch.randn([1, 1]).numpy())
    # signal = signal.reshape([1, -1])
    # plt.plot(signal.cpu().clone().detach().numpy()[0])
    # plt.savefig('./figure.png')
