#!/usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm

from deepmhcii.data_utils import ACIDS, BA_TYPE, EL_TYPE
from deepmhcii.evaluation import CUTOFF

__all__ = ['MHCIIDataset']
ACIDS_VOCAB = ACIDS


class MHCIIDataset(Dataset):
    """

    """
    def __init__(self, data_list, peptide_len=20, peptide_pad=3, mhc_len=34, padding_idx=0):
        self.mhc_names, self.peptide_x, self.mhc_x, self.targets = [], [], [], []
        for mhc_name, peptide_seq, mhc_seq, score in tqdm(data_list, leave=False):
            self.mhc_names.append(mhc_name)
            peptide_x = [ACIDS.index(x if x in ACIDS else '-') for x in peptide_seq][:peptide_len]
            self.peptide_x.append([padding_idx] * peptide_pad +
                                  peptide_x + [padding_idx] * (peptide_len - len(peptide_x)) +
                                  [padding_idx] * peptide_pad)
            assert len(self.peptide_x[-1]) == peptide_len + peptide_pad * 2
            self.mhc_x.append([ACIDS.index(x if x in ACIDS else '-') for x in mhc_seq])
            assert len(self.mhc_x[-1]) == mhc_len
            self.targets.append(score)
        self.peptide_x, self.mhc_x = np.asarray(self.peptide_x), np.asarray(self.mhc_x)
        self.targets = np.asarray(self.targets, dtype=np.float32)

    def __getitem__(self, item):
        return (self.peptide_x[item], self.mhc_x[item]), self.targets[item]

    def __len__(self):
        return len(self.mhc_names)


class ELMHCDataset(Dataset):
    def __init__(self, data_list, mhc_name_seq, peptide_len=15, peptide_pad=3, mhc_len=34, is_sa=False, data_type="EL", thresholds=None, **kwargs):
        super(ELMHCDataset, self).__init__(**kwargs)
        self.cell_line_name, self.peptide_esm_x, self.peptide_x, self.context_x, self.cell_line, self.data_type, self.targets = [], [], [], [], [], [], []
        self.mhc_name_idx = {x: i for i, x in enumerate(mhc_name_seq)}
        for cell_line_name, peptide_seq, context_seq, mhc_name, score in tqdm(data_list, leave=False, miniters=int(len(data_list)/10)+1, maxinterval=360000):
            self.cell_line_name.append(cell_line_name)
            pep_esm_x, pep_emb_x = self.encode_peptide(peptide_seq, peptide_len, peptide_pad)
            self.peptide_esm_x.append(pep_esm_x)
            self.peptide_x.append(pep_emb_x)
            _, context_emb = self.encode_peptide(context_seq, 12, 0)  # Context Sequence
            self.context_x.append(context_emb)
            self.cell_line.append(np.asarray([self.mhc_name_idx[x] for x in mhc_name]))
            self.data_type.append(BA_TYPE if data_type == "BA" else EL_TYPE)
            self.targets.append(score)

        if thresholds != None:
            thresholds_ba, thresholds_el = thresholds
            mhc_thresholds_interval = [[thresholds_ba[mhc_name_seq[x]][0], thresholds_el[mhc_name_seq[x]][0]] for i, x in enumerate(mhc_name_seq)]
            mhc_thresholds_parameter = [[thresholds_ba[mhc_name_seq[x]][1], thresholds_el[mhc_name_seq[x]][1]] for i, x in enumerate(mhc_name_seq)]
            self.mhc_thresholds = np.concatenate([np.asarray(mhc_thresholds_interval)[:,:,1:,None], np.asarray(mhc_thresholds_parameter)],axis=-1)
        else:
            self.mhc_thresholds = np.asarray([[0]]*len(self.mhc_name_idx))  # dummy setting
            
        self.mhc_x = [self.encode_mhc(mhc_name_seq[n_]) for n_ in self.mhc_name_idx]
        self.peptide_esm_x, self.peptide_x, self.context_x, self.mhc_x = \
            np.asarray(self.peptide_esm_x), np.asarray(self.peptide_x), np.asarray(self.context_x), np.asarray(self.mhc_x)
        self.data_type = np.asarray(self.data_type, dtype=np.float32)
        self.targets = np.asarray(self.targets, dtype=np.float32)
        self.is_sa, self.sa_item = is_sa, [i for i in range(len(self.cell_line)) if len(self.cell_line[i]) == 1]
        
    def __getitem__(self, item):
        if self.is_sa:
            item = self.sa_item[item]
        return (self.peptide_x[None, item].repeat(len(c_:=self.cell_line[item]), axis=0), 
                self.mhc_thresholds[c_],
                self.context_x[None, item].repeat(len(c_), axis=0), 
                self.mhc_x[c_],
                len(c_), 
                self.data_type[item],
                self.targets[item])

    def __len__(self):
        return len(self.cell_line_name) if not self.is_sa else len(self.sa_item)
    
    @staticmethod
    def collate_fn(batch):
        peptide_x, peptide_esm_x, context_x, mhc_x, bags_size, data_type, targets = [torch.as_tensor(np.vstack(x)) for x in zip(*batch)]
        return (peptide_x, peptide_esm_x, context_x, mhc_x, bags_size.flatten()), data_type.flatten(), targets.flatten()
        
    def encode_peptide(self, peptide_seq, peptide_len, peptide_pad, padding_idx=0):
        peptide_x = [ACIDS.index(x if x in ACIDS else '-') for x in peptide_seq][:peptide_len]
        peptide_x_out =  [padding_idx] * peptide_pad + peptide_x + [padding_idx] * (peptide_len - len(peptide_x)) + [padding_idx] * peptide_pad
        return ([0], peptide_x_out)
    
    def encode_mhc(self, mhc_seq):
        return [ACIDS.index(x if x in ACIDS else '-') for x in mhc_seq]