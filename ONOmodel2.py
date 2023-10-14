import math
import numpy as np
import torch
import torch.nn as nn
from einops import repeat, rearrange
from torch.nn import functional as F
from functools import partial
from typing import Callable
import warnings
import copy
from timm.models.layers import trunc_normal_
# 
from linear_attention_transformer.linear_attention_transformer import SelfAttention as LinearSelfAttention

# pip install performer-pytorch
from performer_pytorch import SelfAttention as PerformerSelfAttention

# pip install nystrom-attention
from nystrom_attention import NystromAttention

# pip install reformer_pytorch
from reformer_pytorch import LSHSelfAttention

# pip install linformer
from linformer import LinformerSelfAttention


ACTIVATION = {'gelu':nn.GELU,'tanh':nn.Tanh,'sigmoid':nn.Sigmoid,'relu':nn.ReLU,'leaky_relu':nn.LeakyReLU(0.1),'softplus':nn.Softplus,'ELU':nn.ELU,'silu':nn.SiLU}

'''
    A simple MLP class, includes at least 2 layers and n hidden layers
'''

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/2, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=421*421):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class MLP(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()

        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers)])
        # self.bns = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])

    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
            # x = self.act(self.bns[i](self.linears[i](x))) + x
        x = self.linear_post(x)
        return x

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class LinearAttention(nn.Module):

    def __init__(self,
                 dim,
                 attn_type,                 # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 use_ln=False
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type
        self.use_ln = use_ln

        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        if attn_type == 'galerkin':
            if not self.use_ln:
                self.k_norm = nn.InstanceNorm1d(dim_head)
                self.v_norm = nn.InstanceNorm1d(dim_head)
            else:
                self.k_norm = nn.LayerNorm(dim_head)
                self.v_norm = nn.LayerNorm(dim_head)

        elif attn_type == 'fourier':
            if not self.use_ln:
                self.q_norm = nn.InstanceNorm1d(dim_head)
                self.k_norm = nn.InstanceNorm1d(dim_head)
            else:
                self.q_norm = nn.LayerNorm(dim_head)
                self.k_norm = nn.LayerNorm(dim_head)

        else:
            raise Exception(f'Unknown attention type {attn_type}')


        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x):
        # padding mask will be in shape [b, n, 1], it will indicates which point are padded and should be ignored
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)


        if self.attn_type == 'galerkin':
            k = self.norm_wrt_domain(k, self.k_norm)
            v = self.norm_wrt_domain(v, self.v_norm)
        else:  # fourier
            q = self.norm_wrt_domain(q, self.q_norm)
            k = self.norm_wrt_domain(k, self.k_norm)


        dots = torch.matmul(k.transpose(-1, -2), v)
        out = torch.matmul(q, dots) * (1./q.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class ONOBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            attention_dropout: float,
            act='gelu',
            mlp_ratio=4,
            orth=False,
            attn_type=None,
            last_layer=False,
            momentum=0.9,
            psi_dim=64,
            out_dim = 1
    ):
        super().__init__()
        self.orth = orth
        self.momentum = momentum
        self.psi_dim = psi_dim
        
        if self.orth:
            self.register_buffer("feature_cov", None)
        else:
            self.bn = nn.BatchNorm1d(psi_dim)
            #self.ln_4 = nn.LayerNorm(psi_dim)
            #self.mlp3 = MLP(hidden_dim, psi_dim, psi_dim, n_layers=0, res=False, act=act)

        self.register_parameter("mu", nn.Parameter(torch.zeros(psi_dim)))            
        self.ln_1 = nn.LayerNorm(hidden_dim)
        if attn_type == 'performer':
            self.Attn = PerformerSelfAttention(hidden_dim, causal = False, heads = num_heads, dropout=dropout, no_projection=True) # this is not preformer now
        elif attn_type == 'nystrom':
            self.Attn = NystromAttention(hidden_dim, heads = num_heads, dim_head =hidden_dim//num_heads, dropout=dropout)
        elif attn_type == 'reformer':
            self.Attn = LSHSelfAttention(hidden_dim, heads = num_heads, bucket_size = 85, n_hashes = 8, causal = False)
        elif attn_type == 'linformer':
            self.Attn = LinformerSelfAttention(hidden_dim, 7225, heads=num_heads, dropout=dropout)
        elif attn_type == 'galerkin':
            self.Attn = LinearAttention(hidden_dim,'galerkin',heads=num_heads,dim_head=(hidden_dim//num_heads),use_ln=True)         
        else:
            self.Attn = LinearSelfAttention(hidden_dim, causal = False, heads = num_heads, dropout=dropout, attn_dropout=attention_dropout)
        self.attn_type = attn_type
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        #self.time_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),nn.Linear(hidden_dim, hidden_dim))
        self.proj = nn.Linear(hidden_dim, psi_dim)
        #self.proj = MLP(hidden_dim, hidden_dim * mlp_ratio, psi_dim, n_layers=0, res=False, act=act)
        self.ln_3 = nn.LayerNorm(hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, out_dim) if last_layer else MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)

    def forward(self, x, fx , T = None):

        #x = self.ln_1(x + self.Attn(x))
        #x = self.ln_2(x + self.mlp(x))

        x = self.Attn(self.ln_1(x)) + x
        x = self.mlp(self.ln_2(x)) + x

        if self.orth:
            x_ = self.proj(x)            
            if self.training:
                batch_cov = torch.einsum("blc, bld->cd", x_, x_) / x_.shape[0] / x_.shape[1]
                with torch.no_grad():
                    if self.feature_cov is None:
                        self.feature_cov = batch_cov
                    else:
                        self.feature_cov.mul_(self.momentum).add_(batch_cov, alpha=1-self.momentum)
            else:
                batch_cov = self.feature_cov
            L = psd_safe_cholesky(batch_cov)
            L_inv_T = L.inverse().transpose(-2, -1)
            x_ = x_ @ L_inv_T
        else:
            x_ = self.proj(x)
            x_ = self.bn(x_.transpose(-2,-1)).transpose(-2,-1)
            #x_ = self.ln_4(x_)
                
        fx = (x_ * torch.nn.functional.softplus(self.mu)) @ (x_.transpose(-2, -1) @ fx) + fx
        fx = self.mlp2(self.ln_3(fx)) 
        
        return x, fx
    
    
class ONOgateBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            attention_dropout: float,
            act='gelu',
            mlp_ratio=4,
            orth=False,
            attn_type=None,
            momentum=0.9,
            psi_dim=64,
    ):
        super().__init__()
        self.orth = orth
        self.momentum = momentum
        self.psi_dim = psi_dim
        
        if self.orth:
            self.register_buffer("feature_cov", None)
            self.register_buffer("feature_covq", None)
        else:
            #self.bn = nn.BatchNorm1d(psi_dim)
            #self.bn_q = nn.BatchNorm1d(psi_dim)
            self.ln_4 = nn.LayerNorm(psi_dim)
            self.ln_5 = nn.LayerNorm(psi_dim)
            #self.mlp3 = MLP(hidden_dim, psi_dim, psi_dim, n_layers=0, res=False, act=act)

        self.register_parameter("mu", nn.Parameter(torch.zeros(psi_dim)))            
        self.ln_1 = nn.LayerNorm(hidden_dim)
        
        if attn_type == 'performer':
            self.Attn = PerformerSelfAttention(hidden_dim, causal = False, heads = num_heads, dropout=dropout, no_projection=True) # this is not preformer now
            #self.Attn_q = PerformerSelfAttention(hidden_dim, causal = False, heads = num_heads, dropout=dropout, no_projection=True) 
        elif attn_type == 'nystrom':
            self.Attn = NystromAttention(hidden_dim, heads = num_heads, dim_head =hidden_dim//num_heads, dropout=dropout)
            #self.Attn_q = NystromAttention(hidden_dim, heads = num_heads, dim_head =hidden_dim//num_heads, dropout=dropout)
        elif attn_type == 'reformer':
            self.Attn = LSHSelfAttention(hidden_dim, heads = num_heads, bucket_size = 85, n_hashes = 8, causal = False)
            #self.Attn_q = LSHSelfAttention(hidden_dim, heads = num_heads, bucket_size = 85, n_hashes = 8, causal = False)
        elif attn_type == 'linformer':
            self.Attn = LinformerSelfAttention(hidden_dim, 7225, heads=num_heads, dropout=dropout)
            #self.Attn_q = LinformerSelfAttention(hidden_dim, 7225, heads=num_heads, dropout=dropout)
        else:
            self.Attn = LinearSelfAttention(hidden_dim, causal = False, heads = num_heads, dropout=dropout, attn_dropout=attention_dropout)
            #self.Attn_q = LinearSelfAttention(hidden_dim, causal = False, heads = num_heads, dropout=dropout, attn_dropout=attention_dropout)
                
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.ln_3 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        self.mlp_q = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)
        self.ln_1q = nn.LayerNorm(hidden_dim)
        self.ln_2q = nn.LayerNorm(hidden_dim)
        #self.time_fc = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),nn.Linear(hidden_dim, hidden_dim))
        self.proj = nn.Linear(hidden_dim,  psi_dim)
        self.proj_q = nn.Linear(hidden_dim,  psi_dim)
        #self.proj = MLP(hidden_dim, hidden_dim * mlp_ratio, psi_dim, n_layers=0, res=False, act=act)
        self.ln_3 = nn.LayerNorm(hidden_dim)
        self.mlp2 = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, n_layers=0, res=False, act=act)

    def forward(self, x, fx , x_query, T = None):

        #x = self.Attn(self.ln_1(x)) + x
        #x = self.mlp(self.ln_2(x)) + x
        x_query = self.Attn(self.ln_1(x_query)) + x_query
        x_query = self.mlp(self.ln_2(x_query)) + x_query      
        
        if self.orth:
            x_ = self.proj(x)
            x_q = self.proj_q(x_query)            
            if self.training:
                #batch_cov = (torch.einsum("blc, bld->cd", x_, x_)+torch.einsum("blc, bld->cd", x_q, x_q)) /(x_.shape[0] * x_.shape[1] + x_q.shape[0]*x_q.shape[1]) 
                batch_cov = torch.einsum("blc, bld->cd", x_, x_) /(x_.shape[0]*x_.shape[1]) 
                batch_covq = torch.einsum("blc, bld->cd", x_q, x_q) /(x_q.shape[0]*x_q.shape[1]) 
                with torch.no_grad():
                    if self.feature_cov is None:
                        self.feature_cov = batch_cov
                        self.feature_covq = batch_covq
                    else:
                        self.feature_cov.mul_(self.momentum).add_(batch_cov, alpha=1-self.momentum)
                        self.feature_covq.mul_(self.momentum).add_(batch_covq, alpha=1-self.momentum)
            else:
                batch_cov = self.feature_cov
                batch_covq = self.feature_covq
            L = psd_safe_cholesky(batch_cov)
            L_inv_T = L.inverse().transpose(-2, -1)
            L_q = psd_safe_cholesky(batch_covq)
            L_inv_Tq = L_q.inverse().transpose(-2, -1)
            x_ = x_ @ L_inv_T
            x_q = x_q @ L_inv_Tq
        else:
            x_ = self.proj(x)
            x_q = self.proj_q(x_query)
            x_ = self.ln_4(x_)
            x_q = self.ln_5(x_q)
                
        fx = (x_q * torch.nn.functional.softplus(self.mu)) @ (x_.transpose(-2, -1) @ fx) 
        fx = self.mlp2(self.ln_3(fx)) 

        return x_query , fx   
    

class ONO2(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 attn_dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 attn_type=None,
                 mlp_ratio=1,
                 orth=False,
                 psi_dim=64,
                 momentum = 0.9,
                 fun_dim = 1,
                 out_dim = 1
        ):
        super(ONO2, self).__init__()
        self.__name__ = 'ONO'
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.preprocess = MLP(fun_dim + space_dim , n_hidden * 2, n_hidden * 2, n_layers=0, res=False, act=act)
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(),nn.Linear(n_hidden, n_hidden))
         
        self.blocks = nn.ModuleList([ONOBlock(num_heads=n_head, hidden_dim=n_hidden, 
                                              dropout=dropout, attention_dropout=attn_dropout,
                                              act=act, attn_type=attn_type, 
                                              mlp_ratio=mlp_ratio, orth=orth, momentum = momentum,
                                              psi_dim=psi_dim,out_dim = out_dim,
                                              last_layer = (_ == n_layers - 1))
                                        for _ in range(n_layers)])
        #self.decode = nn.Linear(n_hidden,out_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx , T=None):
        if fx is not None:
            x = torch.cat((x, fx), -1)
        x, fx = self.preprocess(x).chunk(2, dim=-1)

        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            x = x + Time_emb   
            
        for block in self.blocks:
            x, fx = block(x , fx)                                 

        #fx = self.decode(fx)
        
        return fx


class ONO3(nn.Module):
    def __init__(self,
                 space_dim=1,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0.0,
                 attn_dropout=0.0,
                 n_head=8,
                 Time_Input=False,
                 act='gelu',
                 attn_type=None,
                 mlp_ratio=1,
                 orth=False,
                 psi_dim=64,
                 momentum = 0.9,
                 fun_dim = 1,
                 out_dim = 1,
                 res = 85
        ):
        super(ONO3, self).__init__()
        self.__name__ = 'ONO'
        self.Time_Input = Time_Input
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.preprocess = MLP(fun_dim + space_dim , n_hidden * 2, n_hidden * 2, n_layers=0, res=False, act=act)
        self.preprocess2 = MLP(space_dim , n_hidden , n_hidden , n_layers=0, res=False, act=act)
        self.preprocess3 = MLP(fun_dim + space_dim , n_hidden , n_hidden , n_layers=0, res=False, act=act)
        self.CroAttn = LinearSelfAttention(n_hidden, causal = False, heads = n_head, dropout=dropout, attn_dropout=attn_dropout, receives_context = True) 
        self.ln_1 = nn.LayerNorm(n_hidden)
        self.ln_2 = nn.LayerNorm(n_hidden)
        if Time_Input:
            self.time_fc = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.SiLU(),nn.Linear(n_hidden, n_hidden))
            
        self.emb_module_f = RotaryEmbedding(n_hidden // space_dim, min_freq=1.0/res)
            #self.preprocess = MLP(T_in + space_dim, n_hidden * 2, n_hidden * 2, n_layers=0, res=False, act=act)
            #self.time_process = MLP(2 * n_hidden, 2 * n_hidden, n_hidden, n_layers=0, res=False, act=act)
        #self.preprocess = MLP(1 + space_dim, n_hidden * mlp_ratio, n_hidden * 2, n_layers=0, act=act)
        #else :
            #self.preprocess = nn.Linear(1+space_dim, n_hidden*2)
        self.gateBlock = ONOgateBlock(num_heads=n_head, hidden_dim=n_hidden, 
                                              dropout=dropout, attention_dropout=attn_dropout,
                                              act=act, attn_type=attn_type, 
                                              mlp_ratio=mlp_ratio, orth=orth, momentum = momentum,
                                              psi_dim=psi_dim)
        self.blocks = nn.ModuleList([ONOBlock(num_heads=n_head, hidden_dim=n_hidden, 
                                              dropout=dropout, attention_dropout=attn_dropout,
                                              act=act, attn_type=attn_type, 
                                              mlp_ratio=mlp_ratio, orth=orth, momentum = momentum,
                                              psi_dim=psi_dim,out_dim = out_dim,
                                              last_layer = (_ == n_layers - 2))
                                        for _ in range(n_layers-1)])
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, fx, x_query, T=None):
        
        if self.space_dim == 2 :
            freq_x = self.emb_module_f.forward(x[..., 0], x.device)
            freq_y = self.emb_module_f.forward(x[..., 1], x.device)  
  
        x = torch.cat((x, fx), -1)
        croinput = self.preprocess3(x)      
        x, fx = self.preprocess(x).chunk(2, dim=-1)

        x_query = self.preprocess2(x_query)
        
        if self.space_dim == 2 :            
            croinput = apply_2d_rotary_pos_emb(croinput, freq_x, freq_y)
            #fx = apply_2d_rotary_pos_emb(fx, freq_x, freq_y)
        
        x_query = self.CroAttn(self.ln_1(x_query), context = self.ln_2(croinput)) + x_query             
        
        if T is not None:
            Time_emb = timestep_embedding(T, self.n_hidden).repeat(1, x.shape[1], 1)
            Time_emb = self.time_fc(Time_emb)
            x = x + Time_emb 
             
        x_query ,fx = self.gateBlock(x , fx , x_query)               
            
        for block in self.blocks:
            x_query, fx = block(x_query , fx) 
                               
        return fx


def psd_safe_cholesky(A, upper=False, out=None, jitter=None):
	"""Compute the Cholesky decomposition of A. If A is only p.s.d, add a small jitter to the diagonal.
	Args:
		:attr:`A` (Tensor):
			The tensor to compute the Cholesky decomposition of
		:attr:`upper` (bool, optional):
			See torch.cholesky
		:attr:`out` (Tensor, optional):
			See torch.cholesky
		:attr:`jitter` (float, optional):
			The jitter to add to the diagonal of A in case A is only p.s.d. If omitted, chosen
			as 1e-6 (float) or 1e-8 (double)
	"""
	try:
		L = torch.linalg.cholesky(A, upper=upper, out=out)
		return L
	except RuntimeError as e:
		isnan = torch.isnan(A)
		if isnan.any():
			raise NanError(
				f"cholesky_cpu: {isnan.sum().item()} of {A.numel()} elements of the {A.shape} tensor are NaN."
			)

		if jitter is None:
			jitter = 1e-6 if A.dtype == torch.float32 else 1e-8
		Aprime = A.clone()
		jitter_prev = 0
		for i in range(10):
			jitter_new = jitter * (10 ** i)
			Aprime.diagonal(dim1=-2, dim2=-1).add_(jitter_new - jitter_prev)
			jitter_prev = jitter_new
			try:
				L = torch.linalg.cholesky(Aprime, upper=upper, out=out)
				warnings.warn(
					f"A not p.d., added jitter of {jitter_new} to the diagonal",
					RuntimeWarning,
				)
				return L
			except RuntimeError:
				continue
		# return torch.randn_like(A).tril()
		raise e
