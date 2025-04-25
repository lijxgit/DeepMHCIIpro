#!/usr/bin/env python3
# -*- coding: utf-8

BA_TYPE, EL_TYPE = 0, 1
ACIDS = '0-ACDEFGHIKLMNPQRSTVWY'
pseudo34_pos = ["A9", "A11", "A22", "A24", "A31", "A52", "A53", "A58", "A59", "A61", "A65", "A66", "A68", "A72", "A73", \
    "B9", "B11", "B13", "B26", "B28", "B30", "B47", "B57", "B67", "B70", "B71", "B74", "B77", "B78", "B81", "B85", "B86", "B89", "B90"]

def get_mhc_name_seq(mhc_name_seq_file):
    mhc_name_seq = {}
    with open(mhc_name_seq_file) as fp:
        for line in fp:
            mhc_name, mhc_seq = line.split()
            mhc_name_seq[mhc_name] = mhc_seq
    return mhc_name_seq


def get_data(data_file, mhc_name_seq):
    data_list = []
    with open(data_file) as fp:
        for line in fp:
            peptide_seq, score, mhc_name = line.split()
            if len(peptide_seq) >= 9:
                data_list.append((mhc_name, peptide_seq, mhc_name_seq[mhc_name], float(score)))
    return data_list

def get_cell_line(allele_list):
    cell_line = {}
    with open(allele_list) as fp:
        for line in fp:
            c_, mhc_name = line.strip().split(' ')
            cell_line[c_] = mhc_name.split(',')
    return cell_line
    
def get_inp_data(data_file, mhc_name_seq=None, allele_list=None, with_index=True, allele_inp=None, use_context=False, model_name=None):
    if allele_list:
        cell_allele_dict = get_cell_line(allele_list)
    if mhc_name_seq:
        for i in mhc_name_seq:
            cell_allele_dict[i] = [i]
    data_list = []
    with open(data_file) as fp:
        for _, line in enumerate(fp):
            line_data = line.split()
            if len(line_data) >= 4 and allele_inp == None:
                peptide_seq, score, cell_line_name, context, *_ = line.split()
            else:
                if allele_inp != None:
                    if len(line_data) >= 4:
                        peptide_seq, score, cell_line_name, context, *_ = line.split()
                    elif len(line_data) >= 3:
                        peptide_seq, score, context, *_ = line.split()
                    elif len(line_data) >= 2:
                        peptide_seq, context, *_ = line.split()
                        score = 0
                    else:
                        print("Providing the data of peptide and score at least")
                    cell_line_name = allele_inp
                    if allele_inp not in cell_allele_dict:
                        cell_allele_dict[allele_inp] = allele_inp.replace(" ","").split(",")
                else:
                    print("Specifying a MHC allele for calculation using the -a or --allele")
                    
            if context != "XXXXXXXXXXXX" and model_name!=None and "Mix" not in model_name:
                context = '0'*(len(context)-len(context.lstrip('X')))+context.strip('X')+'0'*(len(context)-len(context.rstrip('X')))            
            if not use_context:
                context = "XXXXXXXXXXXX"
            assert len(context) == 12
            if len(peptide_seq) < 10:
                peptide_seq = (9 - len(peptide_seq)) * "0" + peptide_seq + (9 - len(peptide_seq)) * "X"
            if allele_list:
                data_list.append((cell_line_name, peptide_seq.upper(), context, cell_allele_dict[cell_line_name], float(score)))
            else:
                data_list.append((cell_line_name, peptide_seq.upper(), context, [cell_line_name], float(score)))
    return data_list

