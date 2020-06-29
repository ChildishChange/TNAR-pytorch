from __future__ import print_function

import argparse
import os

import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image

from vae import VAE
from vgg import vgg11_bn

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--batch-size-ul', type=int, default=128)
parser.add_argument('--coef-vat1', type=float, default=1.0)
parser.add_argument('--coef-vat2', type=float, default=1.0)
parser.add_argument('--coef-ent', type=float, default=1.0)
parser.add_argument('--zeta', type=float, default=0.001)
parser.add_argument('--epsilon1', type=float, default=5.0)
parser.add_argument('--epsilon2', type=float, default=1.0)
parser.add_argument('--resume', type=str, default='./logs/vae')
parser.add_argument('--latent-dim', type=int, default=512)
parser.add_argument('--cuda',type=bool,default=False)
parser.add_argument('--datadir', type=str, default='./TNAR/seed123')
parser.add_argument('--logdir', type=str, default='./logs/tnar')
parser.add_argument('--resultdir', type=str, default='./result/tnar')

args = parser.parse_args()
args.cuda = args.cuda and torch.cuda.is_available()
print(args)
if not os.path.exists(args.logdir):os.makedirs(args.logdir)
if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)

# set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# load dataset
labeled_train_set = np.load(args.datadir+'/labeled_train.npz')
unlabeled_train_set = np.load(args.datadir+'/unlabeled_train.npz')
valid_set = np.load(args.datadir+'/valid.npz')

X_train = torch.from_numpy(labeled_train_set['image'].reshape(-1,32,32,3)).permute(0,3,2,1)
Xul_train = torch.from_numpy(unlabeled_train_set['image'].reshape(-1,32,32,3)).permute(0,3,2,1)
X_valid = torch.from_numpy(valid_set['image'].reshape(-1,32,32,3)).permute(0,3,2,1)

Y_train = torch.from_numpy((np.eye(10)[labeled_train_set['label']]).astype(np.float32))
Y_valid = torch.from_numpy((np.eye(10)[valid_set['label']]).astype(np.float32))

X_loader = DataLoader(TensorDataset(X_train,Y_train),batch_size=args.batch_size,shuffle=False)
X_ul_loader = DataLoader(TensorDataset(Xul_train),batch_size=args.batch_size_ul,shuffle=False)
Valid_loader = DataLoader(TensorDataset(X_valid,Y_valid),batch_size=args.batch_size,shuffle=False)

vae = VAE(args.latent_dim)
vae.load_state_dict(torch.load(args.resume+'/model-120.pkl'))
net = vgg11_bn()

device = torch.device("cuda" if args.cuda else "cpu")  

vae = vae.to(device)
net = net.to(device)


optimizer = optim.Adam(lr = args.lr,params=net.parameters())

def crossentropy(label, logits):
    return -(label* (logits+1e-8).log()).sum(dim=1).mean()
    
def kldivergence(label, logits):
    return (label*((label+1e-8).log()-(logits+1e-8).log())).sum(dim=1).mean()

def normalizevector(r):
    shape = r.size()
    r = r.view(r.size(0),-1)
    r /= (1e-12+r.abs().max(dim=1,keepdim=True)[0])
    r = r.reshape(shape)
    return r

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
    r_adv = normalizevector(torch.randn_like(z))
    r_adv.requires_grad_(True)
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
    
    r_adv_orth = normalizevector(torch.randn_like(x_ul_raw))
    r_adv_orth.requires_grad_(True)
    r_adv_orth1 = 1e-6*r_adv_orth
    
    out_r = net(x_ul_raw+r_adv_orth1)
    
    kl = kldivergence(out_ul, out_r)
    
    r_adv_orth1 = torch.autograd.grad(kl,r_adv_orth1)[0] / 1e-6

    r_adv_orth = r_adv_orth1 - args.zeta*((r_x*r_adv_orth).sum(dim=[1,2,3],keepdim=True)*r_x) + args.zeta*r_adv_orth
    r_adv_orth = normalizevector(r_adv_orth).detach()
    

    return r_adv_orth

#torch.autograd.set_detect_anomaly(True)

# train
for e in range(1,args.epoch+1):
    TOTAL_LOSS, CE_LOSS, VAT_LOSS, VAT_LOSS_ORTH, EN_LOSS = 0,0,0,0,0
    net.train()
    for [x_raw,y],[x_ul_raw] in zip(X_loader,X_ul_loader):
        if args.cuda:
            x_raw, y= x_raw.cuda(),y.cuda()
            x_ul_raw = x_ul_raw.cuda()
        
        x_raw, y= Variable(x_raw), Variable(y)
        x_ul_raw = Variable(x_ul_raw)

        out, out_ul = net(x_raw), net(x_ul_raw)
        mu,logvar = vae.encode(x_ul_raw)
        z = vae.reparameterize(mu,logvar)   
        
        # 计算切向扰动
        r_x = r_vat(z,x_ul_raw,out_ul)
        out_adv = net(x_ul_raw+r_x)
        
        # 计算法向扰动
        r_adv_orth = r_vat_orth(x_ul_raw, r_x, out_ul)
        # torch.cuda.empty_cache()
        out_adv_orth = net(x_ul_raw+r_adv_orth*args.epsilon2)

        # loss 共分为 4 部分
        # vat_loss 只需要算对 out_adv 的梯度， vat_loss_orth 同理
        optimizer.zero_grad()
        vat_loss = kldivergence(out_ul.detach(), out_adv)
        vat_loss_orth = kldivergence(out_ul.detach(), out_adv_orth)
        en_loss = crossentropy(out_ul, out_ul)
        ce_loss = crossentropy(y, out)
        
        VAT_LOSS += args.coef_vat1*vat_loss.item()
        VAT_LOSS_ORTH += args.coef_vat2*vat_loss_orth.item()
        EN_LOSS += args.coef_ent*en_loss.item()
        CE_LOSS += ce_loss.item()
        # ce_loss.backward()
        # total_loss = ce_loss + args.coef_vat1*vat_loss + args.coef_vat2*vat_loss_orth + args.coef_ent*en_loss
        total_loss = ce_loss
        TOTAL_LOSS += total_loss.item()
        total_loss.backward()
        optimizer.step()

    # 打印 r_x r_adv_orth x_ul_raw 的前十个
    # save_image(torch.cat((r_x[:8],r_adv_orth[:8],x_ul_raw[:8]),dim=0),args.resultdir+'/result_' + str(e) + '.png')
    # torch.cuda.empty_cache()

    # if e % 10 == 0:
    acc_valid = 0
    net.eval()
    with torch.no_grad():
        for im,label in Valid_loader:
            if args.cuda: im,label = im.cuda(),label.cuda()

            im, label = Variable(im),Variable(label)
            im, label = im.float(), label.float()

            out = net(im)
            acc_valid += np.mean((torch.argmax(out,1)==torch.argmax(label,1)).cpu().numpy())

    print('[Epoch:%d][ACC:%f][LOSS:%f][CE:%f][VAT:%f][VAT_ORTH:%f][EN:%f]' %
            (e,acc_valid/len(Valid_loader),
                TOTAL_LOSS/len(X_loader),
                CE_LOSS/len(X_loader),
                VAT_LOSS/len(X_loader),
                VAT_LOSS_ORTH/len(X_loader),
                EN_LOSS/len(X_loader)))
    # print('[Epoch:%d][ACC:%f][LOSS:%f]' %
    #         (e,acc_valid/len(Valid_loader),
    #             CE_LOSS/len(X_loader)))
    # if e % 10 == 0:
    #     torch.save(net.state_dict(), args.logdir+'/vggmodel-'+str(e)+'.pkl')
    #     net.cuda()
    