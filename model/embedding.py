#coding:utf-8
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import utils as nn_utils

VERY_SMALL_NUMBER = 1e-10
VERY_NEG_NUMBER = -100000000000

def use_cuda(var):
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var

class EntityEmbedding(nn.Module):
    def __init__(self, entity_embed, trans_units):
        super(EntityEmbedding, self).__init__()
        self.trans_units = trans_units
        self.entity_embedding = nn.Embedding(num_embeddings = entity_embed.shape[0] + 7, embedding_dim = self.trans_units, padding_idx = 0)
        entity_embed = torch.Tensor(entity_embed)
        
        entity_embed = torch.cat((torch.zeros(7, self.trans_units), entity_embed), 0)
        self.entity_embedding.weight = nn.Parameter(use_cuda(torch.Tensor(entity_embed)))
        self.entity_embedding.weight.requires_grad = True
        self.entity_linear = nn.Linear(in_features = self.trans_units, out_features = self.trans_units)

    def forward(self, entity):
        entity_emb = self.entity_embedding(entity) 
        entity_emb = self.entity_linear(entity_emb)
        return entity_emb



class WordEmbedding(nn.Module):
    def __init__(self, word_embed, embed_units):
        super(WordEmbedding, self).__init__()
        
        self.embed_units = embed_units
        self.word_embedding = nn.Embedding(num_embeddings = word_embed.shape[0], embedding_dim = self.embed_units, padding_idx = 0)
        self.word_embedding.weight = nn.Parameter(use_cuda(torch.Tensor(word_embed)))
        self.word_embedding.weight.requires_grad = True

    def forward(self, query_text):
        return self.word_embedding(query_text)

