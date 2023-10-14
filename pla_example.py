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
parser.add_argument('--n-hidden',type=int, default=64, help='hidden dim of ONO')
parser.add_argument('--n-layers',type=int, default=3, help='layers of ONO')
parser.add_argument('--n-heads',type=int, default=4)
parser.add_argument('--batch-size',type=int, default=8)
parser.add_argument("--use_tb", type=int, default=0, help="Use TensorBoard: 1 for True, 0 for False")
parser.add_argument("--gpu", type=str, default='1', help="GPU index to use")
parser.add_argument("--orth", type=int, default=0)
parser.add_argument("--psi_dim", type=int, default=64)
parser.add_argument('--attn_type',type=str, default=None)
parser.add_argument('--max_grad_norm',type=float, default=None)
parser.add_argument('--downsample',type=int,default=1)
parser.add_argument('--momentum',type=float, default=0.9)
parser.add_argument('--mlp_ratio',type=int, default=1)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import scipy.io as scio
import gc
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from timeit import default_timer
from tqdm import *
from testloss import TestLoss

from ONOmodel2 import ONO2

from torch.utils.tensorboard import SummaryWriter



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



class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()
        
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last  # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling mask
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

def random_collate_fn(batch):

    shuffled_batch = []
    shuffled_u = None
    shuffled_t = None
    shuffled_a = None
    shuffled_pos = None
    for item in batch:
        pos = item[0]
        t = item[1]
        a = item[2]
        u = item[3]

        num_timesteps = t.size(0)
        permuted_indices = torch.randperm(num_timesteps)

        t = t[  permuted_indices]
        u = u[... ,permuted_indices]
        
        if shuffled_t is None:
            shuffled_pos = pos.unsqueeze(0)
            shuffled_t = t.unsqueeze(0)
            shuffled_u = u.unsqueeze(0)
            shuffled_a = a.unsqueeze(0)
        else:
            shuffled_pos = torch.cat((shuffled_pos, pos.unsqueeze(0)),0)
            shuffled_t = torch.cat((shuffled_t,t.unsqueeze(0)),0)
            shuffled_u = torch.cat((shuffled_u,u.unsqueeze(0)),0)
            shuffled_a = torch.cat((shuffled_a,a.unsqueeze(0)),0)
            
    shuffled_batch.append(shuffled_pos)
    shuffled_batch.append(shuffled_t)    
    shuffled_batch.append(shuffled_a)
    shuffled_batch.append(shuffled_u)

    return shuffled_batch

def main():
    DATA_PATH = ''

    N = 987
    ntrain = 900
    ntest = 80
    
    s1 = 101
    s2 = 31
    T = 20
    Deformation = 4
    
    r1 = 1
    r2 = 1
    s1 = int(((s1 - 1) / r1) + 1)
    s2 = int(((s2 - 1) / r2) + 1)

    data = scio.loadmat(DATA_PATH)
    input = torch.tensor(data['input'], dtype = torch.float)
    output = torch.tensor(data['output'], dtype = torch.float).transpose(-2,-1)
    print(input.shape, output.shape)
    x_train = input[:ntrain, ::r1][:, :s1].reshape(ntrain,s1,1).repeat(1,1,s2)
    x_train = x_train.reshape(ntrain,-1,1)
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = y_train.reshape(ntrain, -1, Deformation, T)
    x_test = input[-ntest:, ::r1][:, :s1].reshape(ntest,s1,1).repeat(1,1,s2)
    x_test = x_test.reshape(ntest,-1,1)
    y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = y_test.reshape(ntest,-1,  Deformation, T)
    print(x_train.shape, y_train.shape)

    #x_normalizer = UnitGaussianNormalizer(x_train)
    #y_normalizer = UnitGaussianNormalizer(y_train)
    x_normalizer = UnitTransformer(x_train)
    #y_normalizer = UnitTransformer(y_train)
    # x_normalizer = IdentityTransformer(x_train)
    # y_normalizer = IdentityTransformer(y_train)


    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    #y_train = y_normalizer.encode(y_train)
    
    x_normalizer.cuda()
    #y_normalizer.cuda()

    pos = torch.tensor(np.linspace(0, 1, s1*s2), dtype=torch.float)
    pos = pos.reshape(-1, 1).unsqueeze(0)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)
    
    t = np.linspace(0, 1, T)
    t = torch.tensor(t, dtype=torch.float).unsqueeze(0)
    t_train = t.repeat(ntrain,1)
    t_test = t.repeat(ntest,1)
    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, t_train, x_train, y_train),
                                               batch_size=args.batch_size, shuffle=True, collate_fn = random_collate_fn)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, t_test, x_test, y_test),
                                              batch_size=args.batch_size, shuffle=False)
    
    print("Dataloading is over.")
    
    if args.model in ['ONO', 'ONO2','CGPT']:
        if args.model == 'ONO2':
            model = ONO2(n_hidden=args.n_hidden, n_layers=args.n_layers, space_dim=1, fun_dim = 1, out_dim = Deformation, Time_Input=True, n_head = args.n_heads, momentum=args.momentum, orth=args.orth, psi_dim=args.psi_dim, mlp_ratio=args.mlp_ratio, attn_type=args.attn_type).cuda()
        else:
            raise NotImplementedError
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_writer = args.use_tb
    if use_writer:
        writer = SummaryWriter(log_dir='./logs/' + args.model + time.strftime('_%m%d_%H_%M_%S'))
    else:
        writer = None
        
    print(args)
    print(model)
    count_parameters(model)
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs,
                                                    steps_per_epoch=len(train_loader))
    myloss = TestLoss(size_average=False)

    for ep in range(args.epochs):

        model.train()
        train_l2_step = 0

        for x, tim, fx, yy in train_loader:
            loss = 0
            x, fx, tim, yy  = x.cuda(), fx.cuda(), tim.cuda(), yy.cuda() 
            bsz = x.shape[0]    
                    
            for t in range(T):
                y = yy[..., t:t+1]
                input_T = tim[:, t:t+1].reshape(bsz, 1) #B,step
                im = model(x, fx, T = input_T)  

                loss = myloss(im.reshape(bsz,-1), y.reshape(bsz,-1))
                train_l2_step +=  loss.item()
                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
            
            scheduler.step()

        model.eval()
        test_l2_step = 0
        test_l2_full = 0
        with torch.no_grad():
            for x, tim, fx, yy in test_loader:
                loss = 0
                x, fx , tim ,yy  = x.cuda() ,fx.cuda(), tim.cuda(), yy.cuda() 
                bsz = x.shape[0]    
                        
                for t in range(T):
                    y = yy[..., t:t+1]
                    input_T = tim[:, t:t+1].reshape(bsz, 1)
                    #x = torch.cat((x,input_T),-1)
                    im = model(x, fx,T = input_T)
                    loss += myloss(im.reshape(bsz,-1), y.reshape(bsz,-1))
                    #x = x[..., :-1]                   
                    if t == 0:
                        pred = im.unsqueeze(-1)
                    else:
                        pred = torch.cat((pred, im.unsqueeze(-1)), -1)
                    
                test_l2_step +=  loss.item()
                #pred = y_normalizer.decode(pred)
                test_l2_full += myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1)).item()
                
        print("Epoch {} , train_step_loss:{:.5f} , test_step_loss:{:.5f} , test_full_loss:{:.5f}".format(ep,  train_l2_step / ntrain / T , test_l2_step / ntest / T ,test_l2_full / ntest))

        if use_writer:
            writer.add_scalar("train_loss_0", train_l2_step/ntrain/T, ep)
            writer.add_scalar("val loss all", test_l2_full / ntest, ep)

if __name__ == "__main__":
    main()