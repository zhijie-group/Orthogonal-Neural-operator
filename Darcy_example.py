
import os
import time
import argparse
parser = argparse.ArgumentParser('Training Transformer')

parser.add_argument('--lr',type=float, default=1e-3)
parser.add_argument('--epochs',type=int, default=500)
parser.add_argument('--weight_decay',type=float,default=1e-5)
parser.add_argument('--model',type=str, default='ONO2',choices=['FNO','ONO2','ONO'])
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
parser.add_argument('--downsample',type=int,default=5)
parser.add_argument('--momentum',type=float, default=0.9)
parser.add_argument('--mlp_ratio',type=int, default=1)
parser.add_argument('--dropout',type=float, default=0.0)
parser.add_argument('--alpha',type=float, default=0.1)
parser.add_argument('--ntrain',type=int, default=1000)
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
from tqdm import *
from testloss import TestLoss
from einops import repeat, rearrange
from ONOmodel2 import ONO2
from torch.utils.tensorboard import SummaryWriter

train_path = ''
test_path = ''
ntrain = args.ntrain
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


def central_diff(x: torch.Tensor, h, resolution):
    # assuming PBC
    # x: (batch, n, feats), h is the step size, assuming n = h*w
    x = rearrange(x, 'b (h w) c -> b h w c', h=resolution, w=resolution)
    x = F.pad(x,
              (0, 0, 1, 1, 1, 1), mode='constant', value=0.)  # [b c t h+2 w+2]
    grad_x = (x[:, 1:-1, 2:, :] - x[:, 1:-1, :-2, :]) / (2*h)  # f(x+h) - f(x-h) / 2h
    grad_y = (x[:, 2:, 1:-1, :] - x[:, :-2, 1:-1, :]) / (2*h)  # f(x+h) - f(x-h) / 2h

    return grad_x, grad_y

# sol(n,421,421) coeff(n,421,421)
class DarcyDataset(Dataset):
    def __init__(self,
                 data_path=None,
                 n_grid=421,
                 data_len=1000,
                 ):
        self.data_path = data_path
        self.n_grid = n_grid
        self.data_len = data_len

        if self.data_path is not None:
            self._initialize()

    def _initialize(self):
        data = scio.loadmat(self.data_path)

        self.x = data['coeff'][:self.data_len]
        self.x = self.x.reshape(self.x.shape[0], -1, 1)
        self.u = data['sol'][:self.data_len]
        self.u = self.u.reshape(self.u.shape[0], -1, 1)

        del data
        gc.collect()

        x = np.linspace(0, 1, self.n_grid)
        y = np.linspace(0, 1, self.n_grid)
        x, y = np.meshgrid(x, y)
        self.pos = np.c_[x.ravel(), y.ravel()]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        pos = torch.from_numpy(self.pos)
        fx = torch.from_numpy(self.x[index])
        u = torch.from_numpy(self.u[index])

        return dict(pos=pos.float(),
                    fx=fx.float(),
                    u=u.float())




def main():
    r = args.downsample
    h = int(((421 - 1) / r) + 1)
    s = h    
    dx = 1.0/s
    train_data = scio.loadmat(train_path)
    #print(train_data['coeff'].shape)
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

    # x_normalizer = UnitGaussianNormalizer(x_train)
    # y_normalizer = UnitGaussianNormalizer(y_train)
    x_normalizer = UnitTransformer(x_train)
    y_normalizer = UnitTransformer(y_train)
    # x_normalizer = IdentityTransformer(x_train)
    # y_normalizer = IdentityTransformer(y_train)


    x_train = x_normalizer.encode(x_train)
    x_test = x_normalizer.encode(x_test)
    y_train = y_normalizer.encode(y_train)
    
    x_normalizer.cuda()
    y_normalizer.cuda()

    x = np.linspace(0, 1, s)
    y = np.linspace(0, 1, s)
    x, y = np.meshgrid(x, y)
    pos = np.c_[x.ravel(), y.ravel()]
    pos = torch.tensor(pos, dtype=torch.float).unsqueeze(0)
    #pos = pos.repeat(args.batch_size, 1, 1)
    pos_train = pos.repeat(ntrain, 1, 1)
    pos_test = pos.repeat(ntest, 1, 1)

    print("Dataloading is over.")

    
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_train, x_train, y_train),
                                               batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(pos_test, x_test, y_test),
                                              batch_size=args.batch_size, shuffle=False)

    
    if args.model in ['ONO', 'ONO2']:
        if args.model == 'ONO2':
            model = ONO2(n_hidden=args.n_hidden, n_layers=args.n_layers, space_dim=2, fun_dim = 1, dropout = args.dropout, n_head = args.n_heads, momentum=args.momentum, orth=args.orth, psi_dim=args.psi_dim, mlp_ratio=args.mlp_ratio, attn_type=args.attn_type).cuda()
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
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=epochs,steps_per_epoch=len(train_loader))
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs*len(train_loader))
    myloss = TestLoss(size_average=False)
    de_x = TestLoss(size_average=False)
    de_y = TestLoss(size_average=False)
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

            out = rearrange(out.unsqueeze(-1), 'b (h w) c -> b c h w', h=s)
            out = out[..., 1:-1, 1:-1].contiguous()
            out = F.pad(out, (1, 1, 1, 1), "constant", 0)
            out = rearrange(out, 'b c h w -> b (h w) c')
            gt_grad_x, gt_grad_y = central_diff(y.unsqueeze(-1), dx, s)
            pred_grad_x, pred_grad_y = central_diff(out, dx, s)
            deriv_loss = de_x(pred_grad_x, gt_grad_x) + de_y(pred_grad_y, gt_grad_y)            
            loss = args.alpha*deriv_loss + l2loss
            loss.backward()

            # print("loss:{}".format(loss.item()/batch_size))
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_loss += l2loss.item()
            reg += deriv_loss.item()
            scheduler.step()

        train_loss /= ntrain
        reg /= ntrain
        print("Epoch {} Reg : {:.5f} Train loss : {:.5f}".format(ep, reg, train_loss))

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

        if use_writer:
            writer.add_scalar("train_loss_0", train_loss, ep)
            writer.add_scalar("val loss all", rel_err, ep)

if __name__ == "__main__":
    main()