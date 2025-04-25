#!/usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepmhcii.data_utils import ACIDS
from deepmhcii.modules import *
from deepmhcii.init import truncated_normal_

__all__ = ["DeepMHCII_EL_Split_AttMIL",]


class Network(nn.Module):
    """

    """
    def __init__(self, *, emb_size, vocab_size=len(ACIDS), padding_idx=0, peptide_pad=3, mhc_len=34, context_mask=False, **kwargs):
        super(Network, self).__init__()
        self.peptide_emb = nn.Embedding(vocab_size, emb_size)
        self.mhc_emb = nn.Embedding(vocab_size, emb_size)
        self.peptide_pad, self.padding_idx, self.mhc_len, self.emb_size, self.context_mask = peptide_pad, padding_idx, mhc_len, emb_size, context_mask
        
        self.mhc_pos = PositionalEncoding(d_model=self.emb_size, max_len=self.mhc_len)
        self.pep_pos = PositionalEncoding(d_model=self.emb_size, max_len=100)
        
    def forward(self, peptide_x, mhc_x, context_x=None, pair_x=None, *args, **kwargs):
        masks = peptide_x[:, self.peptide_pad: peptide_x.shape[1] - self.peptide_pad] != self.padding_idx
        context_masks = (context_x.unsqueeze(-1)!=0)  if self.context_mask else 1
        
        mhc_x = self.mhc_emb(mhc_x)
        peptide_x = self.peptide_emb(peptide_x)
        mhc_x = self.mhc_pos(mhc_x)
        peptide_x = torch.cat([peptide_x[:, :self.peptide_pad], self.pep_pos(peptide_x[:, self.peptide_pad: peptide_x.shape[1] - self.peptide_pad]), peptide_x[:, peptide_x.shape[1] - self.peptide_pad:]], dim=1)
        
        return peptide_x, mhc_x, masks, None if context_x is None else self.peptide_emb(context_x) * context_masks

    def reset_parameters(self):
        nn.init.uniform_(self.peptide_emb.weight, -0.1, 0.1)
        nn.init.uniform_(self.mhc_emb.weight, -0.1, 0.1)


class DeepMHCII_EL_Split_AttMIL(Network):
    def __init__(self, *, conv_num, conv_size, conv_off, linear_size, dropout=0.5, pooling=True, conv_mask=None, context=False, context_dim=64, rank_pred=False, **kwargs):
        super(DeepMHCII_EL_Split_AttMIL, self).__init__(**kwargs)
        self.pooling, self.conv_masks, self.context, self.rank_pred = pooling, conv_mask, context, rank_pred
        self.conv_off = conv_off
        
        self.conv = nn.ModuleList(IConv(cn, cs, self.mhc_len, expand_dim=True) for cn, cs in zip(conv_num, conv_size))
        self.conv_bn = nn.ModuleList(nn.BatchNorm2d(cn) for cn in conv_num)
        self.conv_R = nn.ModuleList(IConv(cn, cs, self.mhc_len, expand_dim=True) for cn, cs in zip(conv_num, conv_size))
        self.conv_bn_R = nn.ModuleList(nn.BatchNorm2d(cn) for cn in conv_num)
        self.dropout = nn.Dropout(dropout)
        
        self.conv_cum = np.cumsum([0] + conv_num)
        linear_size = [sum(conv_num)] + linear_size
        self.linear = nn.ModuleList([nn.Conv2d(in_s, out_s, 1) for in_s, out_s in zip(linear_size[:-1], linear_size[1:])])
        self.linear_bn = nn.ModuleList([nn.BatchNorm2d(out_s) for out_s in linear_size[1:]])
        self.linear_R = nn.ModuleList([nn.Conv2d(in_s, out_s, 1) for in_s, out_s in zip(linear_size[:-1], linear_size[1:])])
        self.linear_bn_R = nn.ModuleList([nn.BatchNorm2d(out_s) for out_s in linear_size[1:]])
        
        output_indim = linear_size[-1]
        if self.context:
            output_indim = output_indim + context_dim
            self.context_deal = Context_Dealing(self.emb_size, conv_size=context_dim, context_len=12, contex_blocks=4)
            self.dropout_wC = nn.Dropout(dropout)
            
        self.mil_pooling = Att_MIL_Pooling(in_channel=output_indim, hidden_size=output_indim)
        self.output = nn.Conv2d(output_indim, 2, 1)
        
        if conv_mask != None:
            self.conv_masks = torch.Tensor(conv_mask)
        else:
            self.conv_masks = None
        self.reset_parameters()

    def forward(self, peptide_x, peptide_esm_x, context_x, mhc_x, bags_size, pooling=None, inverse=False, with_embed=False, **kwargs):
        peptide_x, mhc_x, masks, context_x = super(DeepMHCII_EL_Split_AttMIL, self).forward(peptide_x, mhc_x, context_x=context_x)
        
        if self.context and context_x != None:
            context_out = self.dropout_wC(self.context_deal(context_x))

        peptide_c = peptide_x.unsqueeze(-1)
        conv_out = torch.cat([F.relu(conv_bn(conv(peptide_c[:, off: peptide_c.shape[1] - off], mhc_x, 
                                                masks=self.conv_masks[off: self.conv_masks.shape[0]-off, :] if self.conv_masks!=None else None)))
                        for conv, conv_bn, off in zip(self.conv, self.conv_bn, self.conv_off)], dim=1)
        conv_out = self.dropout(conv_out)
    
        peptide_i = torch.stack([torch.flip(peptide_x, dims=[1]), peptide_x], dim=-1)
        conv_out_R = torch.cat([F.relu(conv_bn(conv(peptide_i[:, off: peptide_i.shape[1] - off], mhc_x, inversion=True, 
                                                    masks=self.conv_masks[off: self.conv_masks.shape[0]-off, :] if self.conv_masks!=None else None)))
                                    for conv, conv_bn, off in zip(self.conv_R, self.conv_bn_R, self.conv_off)], dim=1)
       
        conv_out_R = torch.cat([torch.flip(conv_out_R[..., [0]], dims=[-2]), conv_out_R[..., [1]]], dim=-1)
        conv_out_R = self.dropout(conv_out_R)
        
        for linear, linear_bn in zip(self.linear, self.linear_bn):
            conv_out = F.relu(linear_bn(linear(conv_out)))
        conv_out = self.dropout(conv_out)
        
        for linear, linear_bn in zip(self.linear_R, self.linear_bn_R):
            conv_out_R = F.relu(linear_bn(linear(conv_out_R)))
        conv_out_R = self.dropout(conv_out_R)
        conv_out = torch.cat([conv_out, conv_out_R], dim=-1)
        
        masks = masks[:, None, -conv_out.shape[2]:, None]

        pool_out, _ = conv_out.masked_fill(~masks, -np.inf).max(dim=2, keepdim=True)
        if self.context and context_x != None:
            pool_out = torch.concat([pool_out, context_out.unsqueeze(-2).repeat(1,1,1,pool_out.shape[-1])], dim=1)
            
        att_out = self.mil_pooling(pool_out, bags_size)["att_out"]
        score_out = torch.sigmoid(self.output(pool_out))
        
        if inverse:
            score_out = torch.stack([torch.sum(score_out[..., :2], dim=-1) * 0.5, score_out[..., -1]], dim=-1)
        else:
            score_out = torch.stack([torch.sum(score_out[..., :2], dim=-1) * 0.5, torch.sum(score_out[..., :2], dim=-1) * 0.5], dim=-1)
        score_out, flag_out = score_out.max(dim=-1)
        offsets = np.cumsum([0] + list(bags_size.cpu()))
        
        att_out = att_out[:, None, None].repeat(1, *score_out.shape[1:])
        if self.rank_pred:
            thresholds_pos = peptide_esm_x.shape[-2] - (peptide_esm_x[..., 0] <= score_out.expand(-1, -1, peptide_esm_x.shape[-2])).sum(-1)
            thresholds_k_b = torch.gather(peptide_esm_x[..., 1:], 2, thresholds_pos[...,None,None].expand(-1,-1,-1,peptide_esm_x.shape[-1]-1))
            rank_out = (score_out - thresholds_k_b[..., 1]) / thresholds_k_b[..., 0]
        
        if not (pooling or self.pooling):
            if self.context and context_x != None:
                conv_out = torch.cat([conv_out, context_out.unsqueeze(-2).repeat(1,1,*conv_out.shape[-2:])], dim=1)
                
            core_score_out = torch.sigmoid(self.output(conv_out).masked_fill(~masks, -np.inf))
            if inverse:
                core_score_out = torch.stack([torch.sum(core_score_out[..., :2], dim=-1) * 0.5, core_score_out[..., -1]], dim=-1)
            else:
                core_score_out = torch.stack([torch.sum(core_score_out[..., :2], dim=-1) * 0.5, torch.sum(core_score_out[..., :2], dim=-1) * 0.5], dim=-1)
            if self.rank_pred:
                core_flag_out = flag_out.repeat(1, 1, core_score_out.shape[-2])
                core_score_out = torch.gather(core_score_out, dim=3, index=core_flag_out[...,None]).squeeze(-1)
                return torch.stack([torch.cat([core_score_out, score_out, rank_out], dim=-1), torch.cat([core_flag_out, flag_out, att_out], dim=-1)], dim=-1).squeeze(1)
            else:
                return core_score_out.squeeze(1).squeeze(-1)
        else:
            if inverse:
                score_out = torch.stack([torch.einsum('b,bld->ld', att_out[i:j], score_out[i:j]) for i,j in zip(offsets[:-1], offsets[1:])])
            else:
                score_out = torch.sum(score_out, dim=-1) * 0.5
            score_out = torch.clamp(score_out, min=0, max=1)
            return torch.cat([score_out, pool_out], dim=1).squeeze(-1) if with_embed else score_out.flatten(start_dim=1)
            

    def reset_parameters(self):
        super(DeepMHCII_EL_Split_AttMIL, self).reset_parameters()
        for conv, conv_bn in zip(self.conv, self.conv_bn):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)
        for linear, linear_bn in zip(self.linear, self.linear_bn):
            truncated_normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)
            linear_bn.reset_parameters()
            nn.init.normal_(linear_bn.weight.data, mean=1.0, std=0.002)
        truncated_normal_(self.output.weight, std=0.1)
        nn.init.zeros_(self.output.bias)
        if self.context:
            self.context_deal.reset_parameters()
        for conv, conv_bn in zip(self.conv_R, self.conv_bn_R):
            conv.reset_parameters()
            conv_bn.reset_parameters()
            nn.init.normal_(conv_bn.weight.data, mean=1.0, std=0.002)
        for linear, linear_bn in zip(self.linear_R, self.linear_bn_R):
            truncated_normal_(linear.weight, std=0.02)
            nn.init.zeros_(linear.bias)
            linear_bn.reset_parameters()
            nn.init.normal_(linear_bn.weight.data, mean=1.0, std=0.002)
