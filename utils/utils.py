#coding:utf-8
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import utils as nn_utils

def padding(sent, l):
    return sent + ['_EOS'] + ['_PAD'] * (l-len(sent)-1)

def padding_triple_id(entity2id, triple, num, l):   
    newtriple = []
    for i in range(len(triple)):
        for j in range(len(triple[i])):
            for k in range(len(triple[i][j])):
                if triple[i][j][k] in entity2id:
                    triple[i][j][k] = entity2id[triple[i][j][k]]
                else:
                    triple[i][j][k] = entity2id['_NONE']

    for tri in triple:
        newtriple.append(tri + [[entity2id['_PAD_H'], entity2id['_PAD_R'], entity2id['_PAD_T']]] * (l - len(tri)))
    pad_triple = [[entity2id['_PAD_H'], entity2id['_PAD_R'], entity2id['_PAD_T']]] * l
    return newtriple + [pad_triple] * (num - len(newtriple))

def build_kb_adj_mat(kb_adj_mats, fact_dropout):
    """Create sparse matrix representation for batched data"""
    mats0_batch = np.array([], dtype=int)
    mats0_0 = np.array([], dtype=int)
    mats0_1 = np.array([], dtype=int)
    vals0 = np.array([], dtype=float)

    mats1_batch = np.array([], dtype=int)
    mats1_0 = np.array([], dtype=int)
    mats1_1 = np.array([], dtype=int)
    vals1 = np.array([], dtype=float)

    for i in range(kb_adj_mats.shape[0]):
        (mat0_0, mat0_1, val0), (mat1_0, mat1_1, val1) = kb_adj_mats[i]
        assert len(val0) == len(val1)
        num_fact = len(val0)
        num_keep_fact = int(np.floor(num_fact * (1 - fact_dropout)))
        mask_index = np.random.permutation(num_fact)[ : num_keep_fact]
        # mat0
        mats0_batch = np.append(mats0_batch, np.full(len(mask_index), i, dtype=int))
        mats0_0 = np.append(mats0_0, mat0_0[mask_index])
        mats0_1 = np.append(mats0_1, mat0_1[mask_index])
        vals0 = np.append(vals0, val0[mask_index])
        # mat1
        mats1_batch = np.append(mats1_batch, np.full(len(mask_index), i, dtype=int))
        mats1_0 = np.append(mats1_0, mat1_0[mask_index])
        mats1_1 = np.append(mats1_1, mat1_1[mask_index])
        vals1 = np.append(vals1, val1[mask_index])

    return (mats0_batch, mats0_0, mats0_1, vals0), (mats1_batch, mats1_0, mats1_1, vals1)
