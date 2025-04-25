#!/usr/bin/env python3
# -*- coding: utf-8

import numpy as np
from collections import namedtuple
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from logzero import logger

__all__ = ['CUTOFF', 'get_auc', 'get_pcc', 'get_srcc', 'get_group_metrics', 'output_res', 'tanh_decay']

CUTOFF = 0.5
Metrics = namedtuple('Metrics', ['auc', 'auc01', 'pcc', 'srcc', 'aupr'])


def tanh_decay(start, end, N_epoch, x):
    assert start > end
    return end + (start - end) * (1 - np.tanh(2 * x / N_epoch))

def get_auc(targets, scores):
    return roc_auc_score(targets >= CUTOFF, scores)

def get_auc01(targets, scores):
    return roc_auc_score(targets >= CUTOFF, scores, max_fpr=0.1)

def get_pcc(targets, scores):
    return np.corrcoef(targets, scores)[0, 1]

def get_srcc(targets, scores):
    return spearmanr(targets, scores)[0]

def get_aupr(targets, scores):
    precision, recall, _ = precision_recall_curve(targets >= CUTOFF, scores)
    return auc(recall, precision)
    
def get_group_metrics(mhc_names, targets, scores, reduce=True, pos_num=3):
    mhc_names, targets, scores = np.asarray(mhc_names), np.asarray(targets), np.asarray(scores)
    mhc_groups, metrics = [], Metrics([], [], [], [], [])
    for mhc_name_ in sorted(set(mhc_names)):
        t_, s_ = targets[mhc_names == mhc_name_], scores[mhc_names == mhc_name_]
        if len(t_[t_ >= CUTOFF]) >= pos_num and len(t_[t_ < CUTOFF]) >= pos_num:
            mhc_groups.append(mhc_name_)
            metrics.auc.append(get_auc(t_, s_))
            metrics.auc01.append(get_auc01(t_, s_))
            metrics.pcc.append(get_pcc(t_, s_))
            metrics.srcc.append(get_srcc(t_, s_))
            metrics.aupr.append(get_aupr(t_, s_))
    return (np.mean(x) for x in metrics) if reduce else (mhc_groups,) + metrics

def get_metrics(mhc_names, targets, scores, pos_num) -> None:
    mhc_names, targets, scores, metrics = np.asarray(mhc_names), np.asarray(targets), np.asarray(scores), []
    mhc_groups, auc, auc01, pcc, srcc, aupr = get_group_metrics(mhc_names, targets, scores, reduce=False, pos_num=pos_num)
    for mhc_name_, auc_, auc01_, pcc_, srcc_, aupr_ in zip(mhc_groups, auc, auc01, pcc, srcc, aupr):
        # t_ = targets[mhc_names == mhc_name_]
        # print(mhc_name_, len(t_), len(t_[t_ >= CUTOFF]), auc_, auc01_, pcc_, srcc_, aupr_)
        metrics.append((auc_, auc01_, pcc_, srcc_, aupr_))
    metrics = np.mean(metrics, axis=0)
    logger.info(f'AUC: {metrics[0]:3f} AUC01: {metrics[1]:3f} PCC: {metrics[2]:3f} SRCC: {metrics[3]:3f} AUPR: {metrics[4]:3f}')
    return metrics