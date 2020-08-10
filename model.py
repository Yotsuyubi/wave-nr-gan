import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from dataset import SignalWithNoise
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



class ResBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.res_block = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(dim, dim, 5, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(dim, dim, 5, padding=2),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3 * output)


class Discriminator(nn.Module):

    def __init__(self, dim, length):
        super().__init__()
        self.dim = dim
        self.length = length
        self.conv = nn.Conv1d(1, self.dim, 1)
        self.block = nn.Sequential(
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
        )
        self.fc = nn.Linear(self.dim*self.length, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        x = nn.Flatten()(x)
        return self.fc(x)

class SignalGenerator(nn.Module):

    def __init__(self, dim, length):
        super().__init__()
        self.dim = dim
        self.length = length
        self.fc = nn.Linear(1, self.length)
        self.conv = nn.Conv1d(1, self.dim, 1)
        self.block = nn.Sequential(
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
        )
        self.out = nn.Conv1d(self.dim, 1, 1)

    def forward(self, noise):
        x = self.fc(noise)
        x = x.reshape(-1, 1, self.length)
        x = self.conv(x)
        x = self.block(x)
        return self.out(x)

class NoiseGenerator(nn.Module):

    def __init__(self, dim, length):
        super().__init__()
        self.dim = dim
        self.length = length
        self.fc = nn.Linear(2, self.length)
        self.conv = nn.Conv1d(1, self.dim, 1)
        self.block = nn.Sequential(
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
            ResBlock(self.dim),
        )
        self.sigma = nn.Linear(self.dim, 1)

    def forward(self, noise, g_latent):
        x = torch.cat([noise, g_latent], axis=1)
        x = self.fc(x)
        x = x.reshape(-1, 1, self.length)
        x = self.conv(x)
        x = self.block(x)
        x = nn.AdaptiveAvgPool1d([1])(x)
        x = nn.Flatten()(x)
        return self.sigma(x)


class GAN(pl.LightningModule):

    def __init__(self, device='cpu'):
        super().__init__()
        self.length = 64
        self.D = Discriminator(256, self.length)
        self.SigG = SignalGenerator(256, self.length)
        self.NoiseG = NoiseGenerator(256, self.length)
        self.dev = device

    def forward(self, x=None):
        if x is not None:
            return self.SigG(x)
        else:
            z = torch.rand([1, 1]).to(self.dev)
            return self.SigG(z)

    def training_step(self, batch, batch_nb, optimizer_idx):

        real = batch.float()

        # train Disc
        if optimizer_idx < 3:

            for p in self.D.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            # train with real
            D_real = self.D(real).mean()

            # train with fake
            z_g = torch.rand([real.size()[0], 1]).to(self.dev)
            z_n = torch.rand([real.size()[0], 1]).to(self.dev)
            fake_signal = self.SigG(z_g)
            noise_sigma = self.NoiseG(z_g, z_n)
            noise_mu = torch.randn([1, self.length]).to(self.dev)
            noise = (noise_sigma * noise_mu).reshape([real.size()[0], 1, -1])
            fake = fake_signal + noise
            D_fake = self.D(fake).mean()

            # train with gradient penalty
            gradient_penalty = self.calc_gradient_penalty(self.D, real, fake)

            D_cost = -D_real+D_fake+gradient_penalty

            return { "loss": D_cost, "progress_bar": { "W_dis": D_real - D_fake } }

        # train Gen
        if optimizer_idx == 3:

            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            # train with converted(fake)
            z_g = torch.rand([real.size()[0], 1]).to(self.dev)
            z_n = torch.rand([real.size()[0], 1]).to(self.dev)
            fake_signal = self.SigG(z_g)
            noise_sigma = self.NoiseG(z_g, z_n)
            noise_mu = torch.randn([1, self.length]).to(self.dev)
            noise = (noise_sigma * noise_mu).reshape([real.size()[0], 1, -1])
            if torch.randn([1]).item() < 0.8:
                fake_signal = fake_signal.clone().detach()
            fake = fake_signal + noise
            C_fake = self.D(fake).mean()
            C_fake_signal = self.D(fake_signal).mean()

            # train with ds reg
            z_n1 = torch.rand([real.size()[0], 1]).to(self.dev)
            z_n2 = torch.rand([real.size()[0], 1]).to(self.dev)
            noise_sigma_1 = self.NoiseG(z_g, z_n1)
            noise_sigma_2 = self.NoiseG(z_g, z_n2)
            ds_reg = torch.min([(noise_sigma_2 - noise_sigma_1).mean() / (z_n2 - z_n1).mean(), 0]).to(self.dev)

            C_cost = -C_fake + ds_reg*0.01

            if batch_nb == 0:
                self.plot()

            return { "loss": C_cost }

    def plot(self):
        signal = self()
        plt.plot(signal.cpu().clone().detach().numpy()[0][0])
        plt.savefig('./figure.png')
        plt.close()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        BATCH_SIZE, _, _ = real_data.size()
        alpha = torch.rand(BATCH_SIZE, 1, 1)
        alpha = alpha.expand(real_data.size()).to(self.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data).to(self.dev)

        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        # TODO: Make ConvBackward diffentiable
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.dev),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def configure_optimizers(self):
        G_params = list(self.SigG.parameters()) + list(self.NoiseG.parameters())
        opt_g = torch.optim.Adam(G_params, lr=1e-4)
        opt_d = torch.optim.Adam(self.D.parameters(), lr=1e-3)
        return [opt_d, opt_d, opt_d, opt_g], []

    def train_dataloader(self):
        return DataLoader(
            SignalWithNoise(self.length, length=1024),
            batch_size=128,
        )
