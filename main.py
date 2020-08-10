from model import GAN
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    gpus = 1 if torch.cuda.is_available() else 0
    device = 'cuda:01' if torch.cuda.is_available() else 'cpu'

    model = GAN(device=device)
    trainer = pl.Trainer(
        gpus=gpus,
        progress_bar_refresh_rate=20,
        max_epochs=1000,
        # auto_lr_find=True
    )
    trainer.fit(model)
