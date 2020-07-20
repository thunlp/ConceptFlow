#coding:utf-8
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import utils as nn_utils
from .embedding import WordEmbedding, EntityEmbedding, use_cuda, VERY_SMALL_NUMBER, VERY_NEG_NUMBER

class OuterEncoder(nn.Module):
    def __init__(self, trans_units, entity_embedding):
        super(OuterEncoder, self).__init__()
        self.EntityEmbedding = entity_embedding
        self.trans_units = trans_units

        self.head_tail_linear = nn.Linear(in_features = self.trans_units * 2, out_features = self.trans_units)
        self.one_two_entity_linear = nn.Linear(in_features = self.trans_units, out_features = self.trans_units)
        self.softmax_d2 = nn.Softmax(dim = 2)

    def forward(self, batch_size, one_two_triples_id, one_two_triple_num):
        one_two_triples_embedding = self.EntityEmbedding(one_two_triples_id).reshape([batch_size, one_two_triple_num, -1, 3 * self.trans_units])
        
        head, relation, tail = torch.split(one_two_triples_embedding, [self.trans_units] * 3, 3)
        head_tail = torch.cat((head, tail), 3)
        head_tail_transformed = torch.tanh(self.head_tail_linear(head_tail)) 

        relation_transformed = self.one_two_entity_linear(relation)

        e_weight = torch.sum(relation_transformed * head_tail_transformed, 3)
        alpha_weight = self.softmax_d2(e_weight)
        
        one_two_embed = torch.sum(alpha_weight.unsqueeze(3) * head_tail, 2)

        return one_two_embed