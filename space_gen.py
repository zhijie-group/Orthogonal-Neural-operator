
import os
import time
import argparse
parser = argparse.ArgumentParser('Training Transformer')

parser.add_argument('--lr',type=float, default=1e-3)
parser.add_argument('--epochs',type=int, default=500)
parser.add_argument('--weight_decay',type=float,default=1e-5)
parser.add_argument('--model',type=str, default='ONO2',choices=['ONO2','ONO'])
parser.add_argument("--modes", type=int, default=12, help="Number of modes")
parser.add_argument("--width", type=int, default=32, help="Width")
parser.add_argument('--n-hidden',type=int, default=128, help='hidden dim of ONO')
parser.add_argument('--n-layers',type=int, default=8, help='layers of ONO')
parser.add_argument('--n-heads',type=int, default=8)
parser.add_argument('--batch-size',type=int, default=8)
parser.add_argument("--use_tb", type=int, default=0, help="Use TensorBoard: 1 for True, 0 for False")
parser.add_argument("--gpu", type=str, default='0', help="GPU index to use")
parser.add_argument("--orth", type=int, default=1)
parser.add_argument("--psi_dim", type=int, default=16)
parser.add_argument('--attn_type',type=str, default=None)
parser.add_argument('--max_grad_norm',type=float, default=0.1)
parser.add_argument('--downsample',type=int,default=10)
parser.add_argument('--momentum',type=float, default=0.9)
parser.add_argument('--mlp_ratio',type=int, default=1)
parser.add_argument('--dropout',type=float, default=0.0)
parser.add_argument('--alpha',type=float, default=0.1)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
from einops import repeat, rearrange
import numpy as np
import scipy.io as scio
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from timeit import default_timer
from testloss import TestLoss
from ONOmodel2 import ONO2

from torch.utils.tensorboard import SummaryWriter

train_path = ''
test_path = ''
ntrain = 1000
ntest = 200
epochs = 500


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params

class IdentityTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True) + 1e-8


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        return x

    def decode(self, x):
        return x


class UnitTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=(0,1), keepdim=True)
        self.std = X.std(dim=(0,1), keepdim=True) + 1e-8


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

    def encode(self, x):
        x = (x - self.mean) / (self.std)
        return x

    def decode(self, x):
        return x* self.std + self.mean


    def transform(self, X, inverse=True,component='all'):
        if component == 'all' or 'all-reduce':
            if inverse:
                orig_shape = X.shape
                return (X*(self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X-self.mean)/self.std
        else:
            if inverse:
                orig_shape = X.shape
                return (X*(self.std[:,component] - 1e-8)+ self.mean[:,component]).view(orig_shape)
            else:
                return (X - self.mean[:,component])/self.std[:,component]

def central_diff(x: torch.Tensor, h, resolution):
    x = rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
    x = F.pad(x,
              (0, 0, 1, 1, 1, 1), mode='constant', value=0.) 
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2*h)  
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2*h)  

    return grad_x, grad_y

class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last  

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                std = self.std[..., sample_idx] + self.eps  # T*batch*n
                mean = self.mean[..., sample_idx]
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# sol(n,421,421) coeff(n,421,421)


def main():
    r = args.downsample
    h = int(((421 - 1) / r) + 1)
    s = h     
    dx = 1.0/s
    print("The resolution of training set is {} * {}".format(s,s))
    train_data = scio.loadmat(train_path)

    x_train = train_data['coeff'][:ntrain, ::r, ::r][:, :s, :s]
    x_train = x_train.reshape(ntrain, -1)
    x_train = torch.from_numpy(x_train).float()
    y_train = train_data['sol'][:ntrain, ::r, ::r][:, :s, :s]
    y_train = y_train.reshape(ntrain, -1)
    y_train = torch.from_numpy(y_train)

    test_data = scio.loadmat(test_path)

    x_test = test_data['coeff'][:ntest, ::r, ::r][:, :s, :s]
    x_test = x_test.reshape(ntest, -1)
    x_test = torch.from_numpy(x_test).float()
    y_test = test_data['sol'][:ntest, ::r, ::r][:, :s, :s]
    y_test = y_test.reshape(ntest, -1)
    y_test = torch.from_numpy(y_test)

    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)
    

    # x_normalizer = UnitGaussianNormalizer(x_train)
    # y_normalizer = UnitGaussianNormalizer(y_train)
    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)
    # x_normalizer = IdentityTransformer(x_train)
    # y_normalizer = IdentityTransformer(y_train)


    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)

    y_train = y_normalizer.encode(y_train)

    sampling_intervals = [7,5,3,2,1]
    gen_dataloaders = []
    for interval in sampling_intervals:
        h_ = int(((421 - 1) / interval) + 1)
        s_ = h_
        x_gen = test_data['coeff'][:ntest, ::interval, ::interval][:, :s_, :s_]
        x_gen = x_gen.reshape(ntest, -1)
        x_gen = torch.from_numpy(x_gen).float()
        y_gen = test_data['sol'][:ntest, ::interval, ::interval][:, :s_, :s_]
        y_gen = y_gen.reshape(ntest, -1)
        y_gen = torch.from_numpy(y_gen)
        x_gen = x_normalizer.encode(x_gen)
        x = np.linspace(0, 1, s_)
        y = np.linspace(0, 1, s_)
        x, y = np.meshgrid(x, y)
        pos = np.c_[x.ravel(), y.ravel()]
        pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
        pos_gen = pos.repeat(ntest, 1, 1)         
        gen_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_gen, x_gen, y_gen),
                                              batch_size=args.batch_size, shuffle=False)
        gen_dataloaders.append(gen_loader)
    
    x_normalizer.cuda()
    y_normalizer.cuda()

    print("Dataloading is over.")

    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test),
                                              batch_size=args.batch_size, shuffle=False)

    

    if args.model in ['ONO', 'ONO2']:
        if args.model == 'ONO2':
            model = ONO2(n_hidden=args.n_hidden, n_layers=args.n_layers, dropout=args.dropout,space_dim=2, fun_dim = 1,n_head = args.n_heads, momentum=args.momentum, orth=args.orth, psi_dim=args.psi_dim, mlp_ratio=args.mlp_ratio, attn_type=args.attn_type).cuda()
        else:
            raise NotImplementedError
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #use_writer = args.use_tb
    #if use_writer:
        #writer = SummaryWriter(log_dir='./logs/' + args.model + time.strftime('_%m%d_%H_%M_%S'))
    #else:
        #writer = None
        
    print(args)
    print(model)
    count_parameters(model)
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    for ep in range(args.epochs):

        model.train()
        train_loss = 0
        reg = 0

        for  x, fx, y in train_loader:

            x, fx , y  = x.cuda() ,fx.cuda(), y.cuda()
            optimizer.zero_grad()

            out = model(x, fx = fx.unsqueeze(-1)).squeeze(-1)    #B, N , 2, fx: B, N, y: B, N

            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)
            
            l2loss = myloss(out, y)                     
            loss = l2loss             
            loss.backward()

            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss += l2loss.item()
            scheduler.step()

        train_loss /= ntrain
        reg /= ntrain
        print("Epoch {} Train loss : {:.5f} Reg : {:.5f}".format(ep, train_loss, reg))

        model.eval()
        rel_err = 0.0

        with torch.no_grad():
            for  x, fx, y in test_loader:
                x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                out = model(x, fx = fx.unsqueeze(-1)).squeeze(-1)
                out = y_normalizer.decode(out)

                tl = myloss(out, y).item()

                rel_err += tl

            rel_err /= ntest
            print("rel_err:{}".format(rel_err))
            
            for gen_loader , r_gen in zip(gen_dataloaders, sampling_intervals):
                gen_err = 0.0                
                for  x, fx, y in gen_loader:
                    x, fx, y = x.cuda(), fx.cuda(), y.cuda()
                    out = model(x, fx = fx.unsqueeze(-1)).squeeze(-1)
                    out = y_normalizer.decode(out)
                    tl = myloss(out, y).item()

                    gen_err += tl

                gen_err /= ntest
                print("s = {} , gen_err :{}".format(int(((421 - 1) / r_gen) + 1) , gen_err))
                

if __name__ == "__main__":
    main()