from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset, TensorDataset

from vae import *

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--augment', type=bool, default=False)
parser.add_argument('--iter-per-epoch', type=int, default=400)
parser.add_argument('--epoch', type=int, default=121)
#parser.add_argument('--lr-decay-epoch', type=int, default=81)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--latent-dim', type=int, default=512)
#parser.add_argument('--kld-coef', type=float, default=0.5)
parser.add_argument('--cuda',type=bool,default=False)
parser.add_argument('--datadir', type=str, default='./TNAR/seed123/')
parser.add_argument('--logdir', type=str, default='./logs/vae_aug')
parser.add_argument('--resultdir', type=str, default='./result/vae_aug')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
print(args)

if not os.path.exists(args.logdir): os.makedirs(args.logdir)
if not os.path.exists(args.datadir): os.makedirs(args.datadir)
if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)

# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# load dataset
unlabeled_train_set = np.load(args.datadir+'/unlabeled_train.npz')
Xul_train = torch.from_numpy(unlabeled_train_set['image'].reshape(-1,32,32,3)).permute(0,3,2,1)
X_ul_loader = DataLoader(TensorDataset(Xul_train),batch_size=args.batch_size,shuffle=True,**kwargs)


def train(model,train_loader,optimizer,device,epoch):
    model.train()
    train_loss = 0
    for batch_idx, [data] in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))

if __name__ == "__main__":
    device = torch.device("cuda" if args.cuda else "cpu")    
    model = VAE(args.latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for e in range(1, args.epoch + 1):
        train(model=model,train_loader=X_ul_loader,optimizer=optimizer,device = device,epoch=e)
        with torch.no_grad():
            sample = torch.randn(64, args.latent_dim).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 32, 32),args.resultdir+'/sample_' + str(e) + '.png')
