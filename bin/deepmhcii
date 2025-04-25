#!/usr/bin/env python3
# -*- coding: utf-8

import click
import numpy as np
from functools import partial
from pathlib import Path
from ruamel.yaml import YAML
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import trange

import json
import deepmhc
from deepmhc.configure import *
from deepmhc.data import MHC_Data_Dir
from deepmhc.models import MHCII_Model_Dir
from deepmhcii.data_utils import BA_TYPE, EL_TYPE, pseudo34_pos

from deepmhcii.data_utils import *
from deepmhcii.datasets import MHCIIDataset, ELMHCDataset
from deepmhcii.models import Model
from deepmhcii.networks import *
from deepmhcii.evaluation import get_metrics, CUTOFF



def run_model(model, model_path, data_list, data_loader, output_path, bi_s_, 
              mode, start_id, num_models, reverse, sort, advanced, max_pool, verbose):
    res_out_list = []
    for model_id in trange(start_id, start_id + num_models):
        model.set_model_path(model_path.with_stem(f'{model_path.stem}-{model_id}'))
        res_out_list.append(model.predict(data_loader, inverse=reverse))
    res_out_list = list(zip(*res_out_list))
    
    el_scores = np.mean(res_out_list[1], axis=0)
    # el_ranks = np.mean(res_out_list[2], axis=0)
    el_att_list = np.stack(res_out_list[0], axis=0)
    el_scores_list = np.stack(res_out_list[1], axis=0)
    el_rank_list = np.stack(res_out_list[2], axis=0)
    ba_att_list = np.stack(res_out_list[3], axis=0)
    ba_scores_list = np.stack(res_out_list[4], axis=0)
    ba_rank_list = np.stack(res_out_list[5], axis=0)
    el_core_idx_list = res_out_list[6]
    el_core_scores_list = res_out_list[7]
    el_bind_mode = np.sum(res_out_list[8], axis=0) > (0.5 * num_models)
     
    lengths = [len(data[-2]) for data in data_list]
    indices = np.cumsum([0] + lengths)
    
    cell_allele_list = [data[-2] for data in data_list]
    cell_allele_score = [el_scores[indices[i]:indices[i+1]] for i in range(len(data_list))]
    cell_bind_mode = [el_bind_mode[indices[i]:indices[i+1]] for i in range(len(data_list))]
    
    best_idxs = np.array([indices[i] + np.argmax(cell_allele_score[i]) for i in range(len(data_list))])
    cell_best_allele = [data_list[i][-2][np.argmax(cell_allele_score[i])] for i in range(len(data_list))]
    
    if mode in ["EL", "Epi"]:
        if max_pool:
            el_cell_line_score, el_cell_line_rank = np.mean(el_scores_list, axis=0), np.mean(el_rank_list, axis=0)
            el_cell_line_score = np.array([np.max(el_cell_line_score[indices[i]:indices[i+1]]) for i in range(len(data_list))])[:, np.newaxis]
            el_cell_line_rank = np.array([np.min(el_cell_line_rank[indices[i]:indices[i+1]]) for i in range(len(data_list))])[:, np.newaxis]
        else:
            el_weighted_scores = el_att_list * el_scores_list
            el_weighted_ranks = el_att_list * el_rank_list
            el_cell_line_score = np.array([np.sum(el_weighted_scores[:, indices[i]:indices[i+1]], axis=1) for i in range(len(data_list))])
            el_cell_line_rank = np.array([np.sum(el_weighted_ranks[:, indices[i]:indices[i+1]], axis=1) for i in range(len(data_list))])

    if mode in ["BA", "Epi"]:
        if max_pool:
            if mode == "Epi":
                ba_scores_list = np.concatenate((ba_scores_list, el_scores_list), axis=0)
                ba_rank_list = np.concatenate((ba_rank_list, el_rank_list), axis=0)
            ba_cell_line_score, ba_cell_line_rank = np.mean(ba_scores_list, axis=0), np.mean(ba_rank_list, axis=0)
            ba_cell_line_score = np.array([np.max(ba_cell_line_score[indices[i]:indices[i+1]]) for i in range(len(data_list))])[:, np.newaxis]
            ba_cell_line_rank = np.array([np.min(ba_cell_line_rank[indices[i]:indices[i+1]]) for i in range(len(data_list))])[:, np.newaxis]
        else:
            ba_weighted_scores = ba_att_list * ba_scores_list
            ba_weighted_ranks = ba_att_list * ba_rank_list
            ba_cell_line_score = np.array([np.sum(ba_weighted_scores[:, indices[i]:indices[i+1]], axis=1) for i in range(len(data_list))])
            ba_cell_line_rank = np.array([np.sum(ba_weighted_ranks[:, indices[i]:indices[i+1]], axis=1) for i in range(len(data_list))])
        
    mode_mapping = {
        "EL": lambda: (np.mean(el_cell_line_score, axis=1), np.mean(el_cell_line_rank, axis=1)),
        "BA": lambda: (np.mean(ba_cell_line_score, axis=1), np.mean(ba_cell_line_rank, axis=1)),
        "Epi": lambda: (np.mean(ba_cell_line_score, axis=1), np.mean(ba_cell_line_rank, axis=1)) \
                        if max_pool else \
                       (np.mean(np.concatenate((ba_cell_line_score, el_cell_line_score), axis=1), axis=1),
                        np.mean(np.concatenate((ba_cell_line_rank, el_cell_line_rank), axis=1), axis=1)) 
    }
    
    mil_scores, ranks = mode_mapping[mode]()
    cell_bind_mode = np.concatenate(cell_bind_mode, axis=0)[best_idxs].astype(int)
    el_core_idx_list = np.stack(el_core_idx_list, axis=0)[:, best_idxs]
    el_core_scores_list = np.stack(el_core_scores_list, axis=0)[:, best_idxs]
    bm_ = (np.sum(el_core_idx_list, axis=0) > (0.5 * len(el_core_idx_list))).astype(int)
    el_idx_masks = el_core_idx_list == bm_[np.newaxis, :, :]
    el_core_scores = np.sum(el_core_scores_list * el_idx_masks, axis=0) / np.sum(el_idx_masks, axis=0)
    s_, p_, ranks = mil_scores, el_core_scores.argmax(axis=1), ranks
    if bi_s_!=None:
        p4_ = bi_s_[0] > bi_s_[1] # zero for P4-specificity
        
    fp = None if output_path == None else open(Path(output_path), 'w')
    if fp != None or verbose:
        cols = ["Num", "Cell_Line", "Peptide", "Best_Allele", "Bind_Core", "Bind_Mode", "Score_{}".format(mode), "%Rank_{}".format(mode)] + \
               (["Bi-Spec"] if bi_s_!=None else []) + \
                (["Allele_Names","Allele_{}_Scores".format(mode)] if advanced  else [])
        len_cols = len(cols)
        table_line = [5, 25, 30, 25, 12, 12, 12, 12, 12, 20, 20]
        formats = ["{:^5}", "{:^25}", "{:^30}", "{:^25}", "{:^12}", "{:^12}", "{:^12}", "{:^12}", "{:^12}", "{:^20}", "{:^20}"]
        assert len(table_line) == len(formats)
        fmt_line = sum(table_line[:len_cols])
        fmt_str = ''.join(formats[:len_cols])   
        
        if output_path!=None and Path(output_path).suffix==".csv":
            print(",".join(cols), file=fp)
        else:
            print("-" * fmt_line, sep="")
            print(fmt_str.format(*cols[:len_cols]), file=fp)
            print("-" * fmt_line, sep="")   
            
        for k in (-s_).argsort()[:int(1*len(s_))] if sort else range(len(s_)):

            piece = [str(k+1), data_list[k][0], data_list[k][2], cell_best_allele[k], data_list[k][2][p_[k]: p_[k] + 9], int(bm_[k][p_[k]]), "{:.6f}".format(s_[k]), "{:.3f}".format(ranks[k])] + \
                    ([p4_[k]] if bi_s_!=None else []) + \
                    ([ ";".join(cell_allele_list[k]), ";".join([str(i) for i in np.round(cell_allele_score[k],3)])] if advanced else [])
            
            if output_path!=None and Path(output_path).suffix==".csv":
                print(",".join([str(i) for i in piece]), file=fp)
            else:
                print(fmt_str.format(*piece), file=fp)
                
        if not(output_path!=None and Path(output_path).suffix==".csv"):
            print("-" * fmt_line, sep="")
            print("Note: Bind_Core and Bind_mode belong to the Best_Allele")
            print("-" * fmt_line, sep="")
    if fp!= None:
        fp.close()

    return (s_, p_, bm_) if bi_s_==None else (s_, p_, bm_, p4_)


@click.command()
@click.option('-i', '--input-path', type=click.Path(exists=True), help='Input file path')
@click.option('-o', '--output-path', type=click.Path(), help='Output file path', default=None)
@click.option('-m', '--mode', type=click.Choice(('BA', 'EL', 'Epi', 'Immu')), default='EL', help="Choose a scoring output among binding affinity, ligand presentation and epitope identification")
@click.option('-w', '--weight-name', type=str, required=True, help='Specified name of model weight')
@click.option('-s', '--start-id', default=0, help="Start id of 25 models for ensemble")
@click.option('-n', '--num-models', default=25, help="End id of 25 models for ensemble")
@click.option('-a', '--allele', default=None, help="Specify allele name and allow multiple alleles, seperated by commas")
@click.option('-c', '--context', is_flag=True, help="Whether to use context information")
@click.option("-r", '--reverse', is_flag=True, help="Whether to consider the reverse binding mode")
@click.option('--motif', type=click.Path(), default=None, help="Save path for generated sequence motif")
@click.option('--mask', type=str, required=False, help="Specify the masked interaction pair, such as B28")
@click.option('--sort', is_flag=True, help="Whether to sort the output scores")
@click.option('--max-pool', is_flag=True, help="Whether to use max-pooling or attention-base multiple instance learning")
@click.option('--advanced', is_flag=True, help="Whether to display score of each allele for multi-allele samples")
@click.option('--evaluation', is_flag=True, help="Whether to evaluate model performance")
@click.option('--verbose', is_flag=True, help="Whether to print output")
def main_process(input_path, output_path, mode, start_id, num_models, allele, context, reverse, sort, mask, advanced, weight_name, max_pool, motif, evaluation, verbose):

    if input_path == None:
        print("You must specify input file by using -i")
        print("Usage: deepmhci [--help] [args] -i inputdir/inputfile")
        print("Test Command: deepmhcii -i {}/MHCII_example.txt".format(MHC_Data_Dir))
        # click.echo(main_process.get_help(click.Context(main_process)))
        return
    input_path = Path(input_path)
    yaml = YAML(typ='safe')
    data_cnf = yaml.load(Path(MHCII_Data_Conf))
    # model_cnf = ModelII_ELContext_Conf if context else ModelII_EL_Conf
    weight_name = weight_name+"-Context" if context else weight_name
    context_model = "Context" in weight_name
    model_cnf = ModelII_ELContext_Conf if context_model else ModelII_EL_Conf
    with open(MHC_Data_Dir + "/" + data_cnf["threshold_wc_ba" if context_model else "threshold_woc_ba"], 'r') as baf, \
         open(MHC_Data_Dir + "/" + data_cnf["threshold_wc_el" if context_model else "threshold_woc_el"], 'r') as elf:
        thresholds = json.load(baf), json.load(elf)
    model_cnf = yaml.load(Path(model_cnf))
    model_cnf["name"] = weight_name
    model_name = model_cnf['name']
    print(f'Model Name: {model_cnf["name"]}')
    model_path = Path(MHCII_Model_Dir)/f'{model_name}.pt'
    mhc_name_seq = get_mhc_name_seq(MHC_Data_Dir+"/"+data_cnf['mhc_seq'])
    allele_list_path  = MHC_Data_Dir+"/"+data_cnf['allele_list']
            
    run_model_fn = partial(run_model, mode=mode, start_id=start_id, num_models=num_models, reverse=reverse, sort=sort, advanced=advanced, max_pool=max_pool, verbose=verbose)
    data_list = get_inp_data(input_path, mhc_name_seq, allele_list=allele_list_path, allele_inp=allele if allele!=None else None, use_context=context, model_name=model_name)
    test_dataset = ELMHCDataset(data_list = data_list, mhc_name_seq=mhc_name_seq, data_type="EL", **model_cnf['padding'], thresholds=thresholds)
    data_loader = DataLoader(test_dataset, collate_fn=ELMHCDataset.collate_fn, batch_size=model_cnf['test']['batch_size'])
    
    bi_s_ = None
    if mask != None:
        bi_s_ = []
        for pep_idx in ["P4", "P6"]:
            pep_idx = int(pep_idx[1])
            mhc_idx = pseudo34_pos.index(mask.upper().strip())
            if model_cnf["model"]["conv_mask"][pep_idx+3-1][mhc_idx] == 0:
                raise("The specified interaction pair cannot work, using the original interaction map")
            else:
                print(f"Masked the inteaction pair of P{pep_idx}-{mask}")
                model_cnf["model"]["conv_mask"][pep_idx+3-1][mhc_idx] = 0
            model = Model(DeepMHCII_EL_Split_AttMIL, model_path=None, pooling=False, rank_pred=True, **model_cnf['model'])
            s_, p_, bm_ = run_model_fn(model, model_path, data_list, data_loader, output_path, bi_s_=None, verbose=False)
            bi_s_.append(s_)
            model_cnf["model"]["conv_mask"][pep_idx+3-1][mhc_idx] = 1

    model = Model(DeepMHCII_EL_Split_AttMIL, model_path=None, pooling=False, rank_pred=True, **model_cnf['model'])
    s_, p_, bm_, *p4_ = run_model_fn(model, model_path, data_list, data_loader, output_path, bi_s_)
    
    if motif != None:
        import logomaker as lm
        import matplotlib.pyplot as plt
        
        bm_ = bm_ if bi_s_ == None else p4_[0]
        spe1_instances, spe2_instances = [], []
        for k in (-s_).argsort()[:int(0.01*len(s_))]:
            bind_mode = int(bm_[k][p_[k]]) if bi_s_ == None  else bm_[k]
            if bind_mode == 0:
                spe1_instances.append(data_list[k][2][p_[k]: p_[k] + 9])
            else:
                spe2_instances.append(data_list[k][2][p_[k]: p_[k] + 9])
        instance_list = [spe1_instances, spe2_instances]
        spe1_frac, spe2_frac = len(spe1_instances) / (0.0001*len(s_)), len(spe2_instances) / (0.0001*len(s_))
        name_list = [f"{allele}-Can({spe1_frac}%)", f"{allele}-Rev({spe2_frac}%)"] if bi_s_ == None \
               else [f"{allele}-P4({spe1_frac}%)", f"{allele}-P6({spe2_frac}%)"]
        instance_list = [i for i in instance_list if len(i)>0]
        name_list = [name_list[i] for i in range(len(instance_list)) if len(instance_list[i])>0]
        fig, axs = plt.subplots(figsize=(22, 8), nrows=1, ncols=2) 
        for data, ax, name in zip(instance_list, axs[:len(instance_list)], name_list):
            counts_mat = lm.alignment_to_matrix(sequences=data, to_type='counts')
            info_mat = lm.transform_matrix(counts_mat, 
                                            from_type='counts', 
                                            to_type='information')
            lm.Logo(info_mat, ax=ax)
            ax.set_title("Binding Motif" if name==None else name, fontsize=24)
            ax.set_xlabel("Position", fontsize=22)
            ax.set_ylabel("Information (bits)", fontsize=22)
            num_positions = info_mat.shape[0]
            ax.set_xticks(range(num_positions))
            ax.set_xticklabels([str(i+1) for i in range(num_positions)], fontsize=20) 
            ax.tick_params(axis="y", labelsize=20)
        for ax in axs[len(instance_list):]:
            fig.delaxes(ax)
            
        plt.savefig(motif, dpi=300, bbox_inches='tight')


    if evaluation:
        mil_scores = s_
        targets_list = [i[-1] for i in data_list]
        group_names = [i[0] for i in data_list]
        if "NEO2019" in input_path.name:
            import pandas as pd
            data = pd.read_csv(input_path, names=["15mer","label","patient","context","pep"], sep="\t")
            data["score"] = mil_scores
            data = data.groupby(["patient", "pep"]).max()
            group_names, targets_list, mil_scores = data.index.get_level_values("patient"), data["label"], data["score"]
        pos_num = 1 if ("NEO2019" in input_path.name or "immun_test" in input_path.name) else 3
        get_metrics(group_names, targets_list, mil_scores, pos_num)
    return
                            
if __name__ == '__main__':
    main_process()
