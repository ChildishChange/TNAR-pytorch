from __future__ import print_function
import os

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

class VAE(nn.Module):
    def __init__(self,latent_dim):
        # 3,32,32/128,15,15/256,6,6
        super(VAE, self).__init__()
        # encode
        self.conv1 = nn.Sequential(nn.Conv2d(3,128,kernel_size=4,stride=2),nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(128,256,kernel_size=5,stride=2),nn.ReLU(inplace=True),)
        self.conv3 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,stride=1),nn.ReLU(inplace=True),)
        self.fc11 = nn.Linear(4*4*512, latent_dim)
        self.fc12 = nn.Linear(4*4*512, latent_dim)
        # decode
        self.fc2 = nn.Sequential(nn.Linear(latent_dim,4*4*512),nn.ReLU(inplace=True))
        self.conv1t = nn.Sequential(nn.ConvTranspose2d(512,256,kernel_size=3,stride=1),nn.ReLU(inplace=True),)
        self.conv2t = nn.Sequential(nn.ConvTranspose2d(256,128,kernel_size=5,stride=2),nn.ReLU(inplace=True),)
        self.conv3t = nn.Sequential(nn.ConvTranspose2d(128,3,kernel_size=4,stride=2))    

    def encode(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        return self.fc11(x), self.fc12(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        x = self.fc2(z)
        x = x.reshape(x.size(0),512,4,4)
        x = self.conv1t(x)
        x = self.conv2t(x)
        return torch.sigmoid(self.conv3t(x))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(recon_x.size(0),-1), x.view(x.size(0),-1), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD
