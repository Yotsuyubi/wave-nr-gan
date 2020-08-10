from model import GAN
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
import numpy as np



if __name__ == '__main__':
    gpus = 1 if torch.cuda.is_available() else 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GAN(device=device)
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=1000,
        # auto_lr_find=True
    )
    trainer.fit(model)
