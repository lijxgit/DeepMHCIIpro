#!/usr/bin/env python3
# -*- coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deepmhcii.init import truncated_normal_

__all__ = ['IConv', 'MIL_Pooling', 'Att_MIL_Pooling', 'Context_Dealing', 'PositionalEncoding']

    
class IConv(nn.Module):
    """

    """
    def __init__(self, out_channels=None, kernel_size=None, mhc_len=34, stride=1, expand_dim= False, **kwargs):
        super(IConv, self).__init__()
        self.mhc_len, self.stride, self.kernel_size, self.expand_dim = mhc_len, stride, kernel_size, expand_dim
        self.weight = nn.Parameter(torch.Tensor(out_channels, kernel_size, mhc_len))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def forward(self, peptide_x, mhc_x, masks=None, inversion=False, **kwargs):
        bs = peptide_x.shape[0]
            
        if masks != None:
            kernel = F.relu(torch.einsum('nld,okl->nodk', mhc_x, self.weight * masks.to(self.weight.device).unsqueeze(0)))
        else:
            kernel = F.relu(torch.einsum('nld,okl->nodk', mhc_x, self.weight))

        if  inversion:
            kernel = torch.flip(kernel, dims=[-1])
            
        if self.expand_dim:
            kernel = kernel.unsqueeze(-1)
            outputs = F.conv2d(peptide_x.transpose(1, 2).reshape(1, -1, peptide_x.shape[1], peptide_x.shape[-1]),
                            kernel.contiguous().view(-1, *kernel.shape[2:]), stride=self.stride, groups=bs)
            return outputs.view(bs, -1, *outputs.shape[-2:]) + self.bias[:, None, None]
        else:
            outputs = F.conv1d(peptide_x.transpose(1, 2).reshape(1, -1, peptide_x.shape[1]),
                            kernel.contiguous().view(-1, *kernel.shape[-2:]), stride=self.stride, groups=bs)
            return outputs.view(bs, -1, outputs.shape[-1]) + self.bias[:, None]

    def reset_parameters(self):
        truncated_normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)


class Context_Dealing(nn.Module):
    """
    
    """
    def __init__(self, emb_size, conv_size, context_len, contex_blocks, **kwargs) -> None:
        super(Context_Dealing, self).__init__()
        block_len = context_len // contex_blocks
        self.cnn_layer = nn.Conv1d(emb_size, conv_size, block_len, block_len)
        n_layers = lambda n: int(np.log2(n))-1 if n > 0 and (n & (n - 1)) == 0 else len(bin(n-1))-3
        linear_size = [contex_blocks*conv_size] + [(2**i)*conv_size for i in range(n_layers(contex_blocks),-1,-1)]
        self.context_linear = nn.ModuleList([nn.Linear(in_s, out_s)
                                     for in_s, out_s in zip(linear_size[:-1], linear_size[1:])])
        self.context_linear_bn = nn.ModuleList([nn.BatchNorm1d(out_s) for out_s in linear_size[1:]])
        
    def forward(self, context):
        context = self.cnn_layer(context.transpose(1,2)).view(context.shape[0], -1)
        for linear, linear_bn in zip(self.context_linear, self.context_linear_bn):
            context = F.relu(linear_bn(linear(context)))
        return context.unsqueeze(-1)
    
    def reset_parameters(self):
        truncated_normal_(self.cnn_layer.weight, std=0.02)
        nn.init.zeros_(self.cnn_layer.bias)
        for linear, linear_bn in zip(self.context_linear, self.context_linear_bn):
            truncated_normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)
            linear_bn.reset_parameters()
            nn.init.normal_(linear_bn.weight.data, mean=1.0, std=0.002)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=34, pos_weight=0.1):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(1, max_len+1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(1000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.ffn = PositionwiseFeedForward(d_model, d_model)
        self.pos_weight = pos_weight
        self.ffn.reset_parameter()

    def forward(self, x):
        x = x + self.pos_weight * self.ffn(self.pe[:x.size(1), :]).unsqueeze(0)
        return x


class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
    
    def forward(self, x):
        x = self.linear2(torch.tanh(self.linear1(x)))
        return x
    
    def reset_parameter(self):
        truncated_normal_(self.linear1.weight, std=0.02)
        nn.init.zeros_(self.linear1.bias)
        truncated_normal_(self.linear2.weight, std=0.02)
        nn.init.zeros_(self.linear2.bias)
    

class MIL_Pooling(nn.Module):
    """
    inputs: sum(bag_size) * kernel_nums matrix \n
    pooling on each bag -> batch_size * kernel_nums \n
    one linear layer project to predictions
    """
    def __init__(self, **kwargs):
        super(MIL_Pooling, self).__init__()

    def forward(self, inter_pre, bags_size, with_inverse=False, score_inter_pre=None, pool_index=None, mean=False):
        offsets = np.cumsum([0] + list(bags_size.cpu()))
        inter_pre_bag = torch.stack([torch.max(inter_pre[i:j,:], dim=0)[0] for i,j in zip(offsets[:-1], offsets[1:])], dim=0)
        return {'debag_out': inter_pre_bag}
        
        
class Att_MIL_Pooling(nn.Module):
    """
        cite from the paper named "Attention-based Deep Multiple Instance Learning", thanks for the author's hard working
    """
    def __init__(self, *, in_channel, hidden_size, **kwargs):
        super(Att_MIL_Pooling, self).__init__()
        self.att_v = nn.Linear(in_channel, hidden_size)
        self.att_u = nn.Linear(in_channel, hidden_size)
        self.att_w = nn.Linear(hidden_size, 1)
        self.reset_parameter()
        
    def forward(self, inter_pre, bags_size):
        inter_pre_, _ = inter_pre.max(dim=-1, keepdim=True)
        inter_pre_ = inter_pre_.contiguous().view(inter_pre_.shape[0], -1)
        inter_v = torch.tanh(self.att_v(inter_pre_))
        inter_u = torch.sigmoid(self.att_u(inter_pre_))
        inter_att = self.att_w(torch.mul(inter_v, inter_u))
        
        offsets = np.cumsum([0] + list(bags_size.cpu()))
        softmax_attn = torch.cat([F.softmax(inter_att[i:j].view(-1), dim=0) for i,j in zip(offsets[:-1], offsets[1:])], dim=0)
        # np.savetxt('tensor.txt', softmax_attn.reshape(-1,9).cpu().numpy(), '%f')
        inter_pre_bag = torch.stack([torch.einsum('b,bld->ld', softmax_attn[i:j], inter_pre.squeeze(-2)[i:j]) for i,j in zip(offsets[:-1], offsets[1:])])
        return {'debag_out': inter_pre_bag.unsqueeze(-2), 'att_out': softmax_attn}
        
    def reset_parameter(self):
        truncated_normal_(self.att_v.weight, std=0.02)
        nn.init.zeros_(self.att_v.bias)
        truncated_normal_(self.att_u.weight, std=0.02)
        nn.init.zeros_(self.att_u.bias)
        truncated_normal_(self.att_w.weight, std=0.02)
        nn.init.zeros_(self.att_w.bias)
