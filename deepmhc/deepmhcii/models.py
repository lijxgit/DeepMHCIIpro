#!/usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from deepmhcii.data_utils import BA_TYPE, EL_TYPE

__all__ = ['Model']


class Model(object):
    """

    """
    def __init__(self, network, model_path, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.network = network(**kwargs, device=self.device).to(self.device)
        if model_path != None:
            self.model_path = Path(model_path)
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.ba_loss_fn, self.el_loss_fn, self.contrast_loss_fn = None, None, None
        self.optimizer, self.scheduler = None, None
        self.training_state = {}
    
        
    def get_scores(self, inputs, embed=False, **kwargs):
        return self.model(*(x.to(self.device) for x in inputs), with_embed=embed, **kwargs)

    @torch.no_grad()
    def predict_step(self, inputs, data_types: torch.Tensor = None, **kwargs):
        self.model.eval()
        if data_types == None:
            return self.get_scores(inputs, **kwargs).cpu()
        else:
            return self.get_scores(inputs, **kwargs)[range(data_types.shape[0]), data_types].cpu()

    def predict(self, data_loader: DataLoader, valid=False, **kwargs):
        if not valid:
            self.load_model()

        res = np.concatenate([self.predict_step(data_x, **kwargs)
                            for data_x, _, _ in tqdm(data_loader, leave=False, miniters=int(len(data_loader)/10)+1, maxinterval=360000)], axis=0)

        el_att_list  = res[..., EL_TYPE, -1, 1]
        el_scores_list  = res[..., EL_TYPE, -2, 0]
        el_rank_list  = res[..., EL_TYPE, -1, 0]
        ba_att_list  = res[..., BA_TYPE, -1, 1]
        ba_scores_list  = res[..., BA_TYPE, -2, 0]
        ba_rank_list  = res[..., BA_TYPE, -1, 0]
        
        el_core_idx_list  = res[..., EL_TYPE, :-2, 1]
        el_core_scores_list  = res[..., EL_TYPE, :-2, 0]
        el_bind_mode  = res[..., EL_TYPE, -2, 1]
        
        return (el_att_list, el_scores_list, el_rank_list, \
                ba_att_list, ba_scores_list, ba_rank_list, \
                el_core_idx_list, el_core_scores_list, el_bind_mode)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        
    def set_model_path(self, model_path):
        self.model_path = model_path
