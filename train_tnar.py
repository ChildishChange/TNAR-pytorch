from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--batch-size-ul', type=int, default=128)
parser.add_argument('--coef-vat1', type=float, default=1.0)
parser.add_argument('--coef-vat2', type=float, default=1.0)
parser.add_argument('--coef-ent', type=float, default=1.0)
parser.add_argument('--zeta', type=float, default=0.001)
parser.add_argument('--epsilon1', type=float, default=5.0)
parser.add_argument('--epsilon2', type=float, default=1.0)
parser.add_argument('--resume', type=str, default='./logs/vae/model-120')
parser.add_argument('--datadir', type=str, default='./CIFAR10/SSL/seed123/')
parser.add_argument('--logdir', type=str, default='./logs/tnar')
args = parser.parse_args()

# load dataset
labeled_train_set = np.load('./labeled_train.npz')
unlabeled_train_set = np.load('./unlabeled_train.npz')
valid_set = np.load('./valid.npz')

X_train = torch.from_numpy(labeled_train_set['image'].reshape(-1,32,32,3)).permute(0,3,2,1)
Xul_train = torch.from_numpy(unlabeled_train_set['image'].reshape(-1,32,32,3)).permute(0,3,2,1)
X_valid = torch.from_numpy(valid_set['image'].reshape(-1,32,32,3)).permute(0,3,2,1)

# 从向量变为对角阵，我也不知道他为什么要变成对角阵，看上去这个数据只有十类
Y_train = torch.from_numpy((np.eye(10)[labeled_train_set['label']]).astype(np.float32))
Y_valid = torch.from_numpy((np.eye(10)[valid_set['label']]).astype(np.float32))

X_loader = DataLoader(TensorDataset(X_train,Y_train),batch_size=args.batch_size,shuffle=True)
X_ul_loader = DataLoader(TensorDataset(Xul_train),batch_size=args.batch_size_ul,shuffle=True)


from vae import VAE
from vgg import vgg11_bn

vae = VAE()
net = vgg11_bn()

optimizer = optim.Adam(lr = args.lr,params=net.parameters())

def crossentropy(label, logits):
    return -(label* (logits+1e-8).log()).sum(dim=1).mean()
    
def kldivergence(label, logits):
    return (label*((label+1e-8).log()-(logits+1e-8).log())).sum(dim=1).mean()

def normalizevector(r):
    shape = r.size()
    r = r.view(r.size(0),-1)
    r /= (1e-12+r.abs().max(dim=1,keepdim=True)[0])
    # r /= (1e-6 + ((r.sum(dim=1,keepdim=True)[0]))**0.5)
    return r.reshape(shape)

def r_vat(z,x_ul_raw,out_ul):
    x_recon = vae.decode(z)
    
    # 
    r0 = Variable(torch.zeros_like(z),requires_grad=True)
    x_recon_r0 = vae.decode(z+r0)

    diff2 = 0.5*((x_recon - x_recon_r0)**2).sum(dim=[1,2,3])

    diffJaco = torch.autograd.grad(outputs = diff2,inputs = r0,
                                    grad_outputs=torch.ones_like(diff2)
                                    ,create_graph=True)[0]
    
    #
    r_adv = normalizevector(torch.randn_like(z,requires_grad=True))
    r_adv = 1e-6*r_adv
    x_r = vae.decode(z+r_adv)
    
    out_r = net(x_r-x_recon+x_ul_raw)
    kl = kldivergence(out_ul, out_r)
    
    r_adv = torch.autograd.grad(kl,r_adv,retain_graph=True)[0] / 1e-6
    r_adv = normalizevector(r_adv)
    
    rk = r_adv + 0
    pk = rk + 0
    xk = torch.zeros_like(rk)
    for k in range(4):
        temp = diffJaco*pk
        all_one = torch.ones_like(temp)
        Bpk = torch.autograd.grad(outputs = temp,
                                  inputs= r0,
                                  grad_outputs = all_one,retain_graph=True)[0].detach()
        
        pkBpk = (pk*Bpk).sum(dim=1,keepdim=True)
        rk2 = (rk*rk).sum(dim=1,keepdim=True)
        alphak = (rk2 / (pkBpk+1e-8)) * (rk2>1e-8).float()
        xk += alphak * pk
        rk -= alphak * Bpk
        betak = (rk*rk).sum(dim=1, keepdim=True) / (rk2+1e-8)
        pk = rk + betak * pk

    r_adv = normalizevector(xk)
    x_adv = vae.decode(z+r_adv*args.epsilon1)
    r_x = x_adv - x_recon
    r_x = normalizevector(r_x).detach()
    return r_x

def r_vat_orth(x_ul_raw, r_x, out_ul):
    
    r_adv_orth = normalizevector(torch.randn_like(x_ul_raw,requires_grad=True))
    r_adv_orth1 = 1e-6*r_adv_orth
    
    out_r = net(x_ul_raw+r_adv_orth1)
    
    kl = kldivergence(out_ul, out_r)
    
    r_adv_orth1 = torch.autograd.grad(kl,r_adv_orth1)[0] / 1e-6

    r_adv_orth = r_adv_orth1 - args.zeta*((r_x*r_adv_orth).sum(dim=[1,2,3],keepdim=True)*r_x) + args.zeta*r_adv_orth
    r_adv_orth = normalizevector(r_adv_orth).detach()
    

    return r_adv_orth

torch.autograd.set_detect_anomaly(True)
# train
for ep in range(args.epoch):
    print(ep)
    for [x_raw,y],x_ul_raw in zip(X_loader,X_ul_loader):
        print("a")
        x_raw, y= Variable(x_raw), Variable(y)
        x_ul_raw = Variable(x_ul_raw[0])

        out, out_ul = net(x_raw), net(x_ul_raw)
        mu,logvar = vae.encode(x_ul_raw)
        z = vae.reparameterize(mu,logvar,False)   
        
        # 计算切向扰动
        r_x = r_vat(z,x_ul_raw,out_ul)
        out_adv = net(x_ul_raw+r_x)
        # 计算法向扰动
        r_adv_orth = r_vat_orth(x_ul_raw, r_x, out_ul)
        out_adv_orth = net(x_ul_raw+r_adv_orth*args.epsilon2)

        # loss 共分为 4 部分
        # vat_loss 只需要算对 out_adv 的梯度， vat_loss_orth 同理
        optimizer.zero_grad()
        vat_loss = kldivergence(out_ul.detach(), out_adv)
        vat_loss_orth = kldivergence(out_ul.detach(), out_adv_orth)
        en_loss = crossentropy(out_ul, out_ul)
        ce_loss = crossentropy(y, out)
        total_loss = ce_loss + args.coef_vat1*vat_loss + args.coef_vat2*vat_loss_orth + args.coef_ent*en_loss

        total_loss.backward()
        optimizer.step()
        
