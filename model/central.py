#coding:utf-8
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import utils as nn_utils
from .embedding import WordEmbedding, EntityEmbedding, use_cuda, VERY_SMALL_NUMBER, VERY_NEG_NUMBER

class CentralEncoder(nn.Module):
    def __init__(self, config, gnn_layers, embed_units, trans_units, word_embedding, entity_embedding):
        super(CentralEncoder, self).__init__()
        self.k = 2 + 1
        self.gnn_layers = gnn_layers
        self.WordEmbedding = word_embedding
        self.EntityEmbedding = entity_embedding
        self.embed_units = embed_units
        self.trans_units = trans_units
        self.pagerank_lambda = config.pagerank_lambda
        self.fact_scale = config.fact_scale

        self.node_encoder = nn.LSTM(input_size = self.embed_units, hidden_size = self.trans_units, batch_first=True, bidirectional=False)
        self.lstm_drop = nn.Dropout(p = config.lstm_dropout)
        self.softmax_d1 = nn.Softmax(dim = 1)
        self.linear_drop = nn.Dropout(p = config.linear_dropout)
        self.relu = nn.ReLU()

        for i in range(self.gnn_layers):
            self.add_module('q2e_linear' + str(i), nn.Linear(in_features=self.trans_units, out_features=self.trans_units))
            self.add_module('d2e_linear' + str(i), nn.Linear(in_features=self.trans_units, out_features=self.trans_units))
            self.add_module('e2q_linear' + str(i), nn.Linear(in_features=self.k * self.trans_units, out_features=self.trans_units))
            self.add_module('e2d_linear' + str(i), nn.Linear(in_features=self.k * self.trans_units, out_features=self.trans_units))
            self.add_module('e2e_linear' + str(i), nn.Linear(in_features=self.k * self.trans_units, out_features=self.trans_units))
            
            #use kb
            self.add_module('kb_head_linear' + str(i), nn.Linear(in_features=self.trans_units, out_features=self.trans_units))
            self.add_module('kb_tail_linear' + str(i), nn.Linear(in_features=self.trans_units, out_features=self.trans_units))
            self.add_module('kb_self_linear' + str(i), nn.Linear(in_features=self.trans_units, out_features=self.trans_units))

    def forward(self, batch_size, max_local_entity, max_fact, query_text, local_entity, q2e_adj_mat, kb_adj_mat, kb_fact_rel, query_mask):
        # normalized adj matrix
        pagerank_f = use_cuda(Variable(torch.from_numpy(q2e_adj_mat).type('torch.FloatTensor'), requires_grad=True)) 
        q2e_adj_mat = use_cuda(Variable(torch.from_numpy(q2e_adj_mat).type('torch.FloatTensor'), requires_grad=False)) 
        assert pagerank_f.requires_grad == True

        # encode query
        query_word_emb = self.WordEmbedding(query_text)
        query_hidden_emb, (query_node_emb, _) = self.node_encoder(self.lstm_drop(query_word_emb), self.init_hidden(1, batch_size, self.trans_units)) 
        query_node_emb = query_node_emb.squeeze(dim=0).unsqueeze(dim=1) 
        query_rel_emb = query_node_emb 

        # build kb_adj_matrix from sparse matrix
        (e2f_batch, e2f_f, e2f_e, e2f_val), (f2e_batch, f2e_e, f2e_f, f2e_val) = kb_adj_mat
        entity2fact_index = torch.LongTensor([e2f_batch, e2f_f, e2f_e])
        entity2fact_val = torch.FloatTensor(e2f_val)
        entity2fact_mat = use_cuda(torch.sparse.FloatTensor(entity2fact_index, entity2fact_val, torch.Size([batch_size, max_fact, max_local_entity]))) 
        fact2entity_index = torch.LongTensor([f2e_batch, f2e_e, f2e_f])
        fact2entity_val = torch.FloatTensor(f2e_val)
        fact2entity_mat = use_cuda(torch.sparse.FloatTensor(fact2entity_index, fact2entity_val, torch.Size([batch_size, max_local_entity, max_fact])))
            
        local_fact_emb = self.EntityEmbedding(kb_fact_rel) 

        # attention fact2question
        div = float(np.sqrt(self.trans_units))
        fact2query_sim = torch.bmm(query_hidden_emb, local_fact_emb.transpose(1, 2)) / div 
        fact2query_sim = self.softmax_d1(fact2query_sim + (1 - query_mask.unsqueeze(dim=2)) * VERY_NEG_NUMBER) 
            
        fact2query_att = torch.sum(fact2query_sim.unsqueeze(dim=3) * query_hidden_emb.unsqueeze(dim=2), dim=1) 
            
        W = torch.sum(fact2query_att * local_fact_emb, dim=2) / div 
        W_max = torch.max(W, dim=1, keepdim=True)[0] 
        W_tilde = torch.exp(W - W_max) 
        e2f_softmax = self.sparse_bmm(entity2fact_mat.transpose(1, 2), W_tilde.unsqueeze(dim=2)).squeeze(dim=2) 
        e2f_softmax = torch.clamp(e2f_softmax, min=VERY_SMALL_NUMBER)
        e2f_out_dim = use_cuda(Variable(torch.sum(entity2fact_mat.to_dense(), dim=1), requires_grad=False)) 
        
        # load entity embedding 
        local_entity_emb = self.EntityEmbedding(local_entity) 
   
        # label propagation on entities
        for i in range(self.gnn_layers):
            # get linear transformation functions for each layer
            q2e_linear = getattr(self, 'q2e_linear' + str(i))
            d2e_linear = getattr(self, 'd2e_linear' + str(i))
            e2q_linear = getattr(self, 'e2q_linear' + str(i))
            e2d_linear = getattr(self, 'e2d_linear' + str(i))
            e2e_linear = getattr(self, 'e2e_linear' + str(i))
          
            kb_self_linear = getattr(self, 'kb_self_linear' + str(i))
            kb_head_linear = getattr(self, 'kb_head_linear' + str(i))
            kb_tail_linear = getattr(self, 'kb_tail_linear' + str(i))

            # start propagation
            next_local_entity_emb = local_entity_emb

            # STEP 1: propagate from question, documents, and facts to entities
            # question -> entity
            q2e_emb = q2e_linear(self.linear_drop(query_node_emb)).expand(batch_size, max_local_entity, self.trans_units) 
            next_local_entity_emb = torch.cat((next_local_entity_emb, q2e_emb), dim=2) 

            # fact -> entity
            e2f_emb = self.relu(kb_self_linear(local_fact_emb) + self.sparse_bmm(entity2fact_mat, kb_head_linear(self.linear_drop(local_entity_emb)))) 
            e2f_softmax_normalized = W_tilde.unsqueeze(dim=2) * self.sparse_bmm(entity2fact_mat, (pagerank_f / e2f_softmax).unsqueeze(dim=2)) 
            e2f_emb = e2f_emb * e2f_softmax_normalized 
            f2e_emb = self.relu(kb_self_linear(local_entity_emb) + self.sparse_bmm(fact2entity_mat, kb_tail_linear(self.linear_drop(e2f_emb))))
                
            pagerank_f = self.pagerank_lambda * self.sparse_bmm(fact2entity_mat, e2f_softmax_normalized).squeeze(dim=2) + (1 - self.pagerank_lambda) * pagerank_f 

            # STEP 2: combine embeddings from fact
            next_local_entity_emb = torch.cat((next_local_entity_emb, self.fact_scale * f2e_emb), dim=2) 
            
            # STEP 3: propagate from entities to update question, documents, and facts
            # entity -> query
            query_node_emb = torch.bmm(pagerank_f.unsqueeze(dim=1), e2q_linear(self.linear_drop(next_local_entity_emb)))
            # update entity
            local_entity_emb = self.relu(e2e_linear(self.linear_drop(next_local_entity_emb))) 

        return local_entity_emb

    def init_hidden(self, num_layer, batch_size, hidden_size):
        return (use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))), 
                use_cuda(Variable(torch.zeros(num_layer, batch_size, hidden_size))))

    def sparse_bmm(self, X, Y):
        """Batch multiply X and Y where X is sparse, Y is dense.
        Args:
            X: Sparse tensor of size BxMxN. Consists of two tensors,
                I:3xZ indices, and V:1xZ values.
            Y: Dense tensor of size BxNxK.
        Returns:
            batched-matmul(X, Y): BxMxK
        """
        class LeftMMFixed(torch.autograd.Function):
            """
            Implementation of matrix multiplication of a Sparse Variable with a Dense Variable, returning a Dense one.
            This is added because there's no autograd for sparse yet. No gradient computed on the sparse weights.
            """

            def __init__(self):
                super(LeftMMFixed, self).__init__()
                self.sparse_weights = None

            def forward(self, sparse_weights, x):
                if self.sparse_weights is None:
                    self.sparse_weights = sparse_weights
                return torch.mm(self.sparse_weights, x)

            def backward(self, grad_output):
                sparse_weights = self.sparse_weights
                return None, torch.mm(sparse_weights.t(), grad_output)

        I = X._indices()
        V = X._values()
        B, M, N = X.size()
        _, _, K = Y.size()
        Z = I.size()[1]
        lookup = Y[I[0, :], I[2, :], :]
        X_I = torch.stack((I[0, :] * M + I[1, :], use_cuda(torch.arange(Z).type(torch.LongTensor))), 0)
        S = use_cuda(Variable(torch.sparse.FloatTensor(X_I, V, torch.Size([B * M, Z])), requires_grad=False))
        prod_op = LeftMMFixed()
        prod = prod_op(S, lookup)
        return prod.view(B, M, K)
