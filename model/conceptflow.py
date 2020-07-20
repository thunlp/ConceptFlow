#coding:utf-8
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from torch.nn import utils as nn_utils
from .central import CentralEncoder
from .outer import OuterEncoder
from .embedding import WordEmbedding, EntityEmbedding, use_cuda, VERY_SMALL_NUMBER, VERY_NEG_NUMBER



class ConceptFlow(nn.Module):
    def __init__(self, config, word_embed, entity_embed, is_select=False):
        super(ConceptFlow, self).__init__()
        self.is_select = is_select
        self.is_inference = False
    
        self.trans_units = config.trans_units 
        self.embed_units = config.embed_units 
        self.units = config.units 
        self.layers = config.layers
        self.gnn_layers = config.gnn_layers
        self.symbols = config.symbols

        self.WordEmbedding = WordEmbedding(word_embed, self.embed_units)
        self.EntityEmbedding = EntityEmbedding(entity_embed, self.trans_units)
        self.CentralEncoder = CentralEncoder(config, self.gnn_layers, self.embed_units, self.trans_units, self.WordEmbedding, self.EntityEmbedding)
        self.OuterEncoder = OuterEncoder(self.trans_units, self.EntityEmbedding)

        self.softmax_d1 = nn.Softmax(dim = 1)
        self.softmax_d2 = nn.Softmax(dim = 2)
        
        self.text_encoder = nn.GRU(input_size = self.embed_units, hidden_size = self.units, num_layers = self.layers, batch_first = True)
        self.decoder = nn.GRU(input_size = self.units + self.embed_units, hidden_size = self.units, num_layers = self.layers, batch_first = True)

        self.attn_c_linear = nn.Linear(in_features = self.units, out_features = self.units, bias = False)
        self.attn_ce_linear = nn.Linear(in_features = self.trans_units, out_features = 2 * self.units, bias = False)
        self.attn_co_linear = nn.Linear(in_features = 2 * self.trans_units, out_features = 2 * self.units, bias = False)
        self.attn_ct_linear = nn.Linear(in_features = self.trans_units, out_features = 2 * self.units, bias = False)

        self.context_linear = nn.Linear(in_features = 4 * self.units, out_features = self.units, bias = False)


        self.logits_linear = nn.Linear(in_features = self.units, out_features = self.symbols)
        self.selector_linear = nn.Linear(in_features = self.units, out_features = 3)

    def forward(self, batch_data):
        query_text = batch_data['query_text']
        answer_text = batch_data['answer_text']
        local_entity = batch_data['local_entity']
        responses_length = batch_data['responses_length']
        q2e_adj_mat = batch_data['q2e_adj_mat']
        kb_adj_mat = batch_data['kb_adj_mat'] 
        kb_fact_rel = batch_data['kb_fact_rel']
        match_entity_one_hop = batch_data['match_entity_one_hop']
        only_two_entity = batch_data['only_two_entity']
        match_entity_only_two = batch_data['match_entity_only_two']
        one_two_triples_id = batch_data['one_two_triples_id']
        local_entity_length = batch_data['local_entity_length']
        only_two_entity_length = batch_data['only_two_entity_length']

        if self.is_inference == True:
            word2id = batch_data['word2id']
            entity2id = batch_data['entity2id']
            id2entity = dict()
            for key in entity2id.keys():
                id2entity[entity2id[key]] = key
        else:
            id2entity = None
 
        batch_size, max_local_entity = local_entity.shape
        _, max_only_two_entity = only_two_entity.shape
        _, one_two_triple_num, one_two_triple_len, _ = one_two_triples_id.shape
        _, max_fact = kb_fact_rel.shape

        # numpy to tensor
        local_entity = use_cuda(Variable(torch.from_numpy(local_entity).type('torch.LongTensor'), requires_grad=False))
        local_entity_mask = use_cuda((local_entity != 0).type('torch.FloatTensor'))
        kb_fact_rel = use_cuda(Variable(torch.from_numpy(kb_fact_rel).type('torch.LongTensor'), requires_grad=False))
        query_text = use_cuda(Variable(torch.from_numpy(query_text).type('torch.LongTensor'), requires_grad=False))
        answer_text = use_cuda(Variable(torch.from_numpy(answer_text).type('torch.LongTensor'), requires_grad=False))
        responses_length = use_cuda(Variable(torch.Tensor(responses_length).type('torch.LongTensor'), requires_grad=False))
        query_mask = use_cuda((query_text != 0).type('torch.FloatTensor'))
        match_entity_one_hop = use_cuda(Variable(torch.from_numpy(match_entity_one_hop).type('torch.LongTensor'), requires_grad=False))
        only_two_entity = use_cuda(Variable(torch.from_numpy(only_two_entity).type('torch.LongTensor'), requires_grad=False))
        match_entity_only_two = use_cuda(Variable(torch.from_numpy(match_entity_only_two).type('torch.LongTensor'), requires_grad=False))
        one_two_triples_id = use_cuda(Variable(torch.from_numpy(one_two_triples_id).type('torch.LongTensor'), requires_grad=False))


        decoder_len = answer_text.shape[1]
        encoder_len = query_text.shape[1]
        responses_target = answer_text
        responses_id = torch.cat((use_cuda(torch.ones([batch_size, 1]).type('torch.LongTensor')),torch.split(answer_text, [decoder_len - 1, 1], 1)[0]), 1)
        

        # encode central graph
        local_entity_emb = self.CentralEncoder(batch_size, max_local_entity, max_fact, query_text, local_entity, q2e_adj_mat, kb_adj_mat, kb_fact_rel, query_mask)
        
        # encode text
        text_encoder_input = self.WordEmbedding(query_text)
        text_encoder_output, text_encoder_state = self.text_encoder(text_encoder_input, use_cuda(Variable(torch.zeros(self.layers, batch_size, self.units))))

        # encode outer graph
        one_two_embed = self.OuterEncoder(batch_size, one_two_triples_id, one_two_triple_num)

        # prepare decoder input for training
        decoder_input = self.WordEmbedding(responses_id)

        # attention key and values
        c_attention_keys = self.attn_c_linear(text_encoder_output)
        c_attention_values = text_encoder_output
        ce_attention_keys, ce_attention_values = torch.split(self.attn_ce_linear(local_entity_emb), [self.units, self.units], 2)
        co_attention_keys, co_attention_values = torch.split(self.attn_co_linear(one_two_embed), [self.units, self.units], 2)
        only_two_entity_embed = self.EntityEmbedding(only_two_entity)
        ct_attention_keys, ct_attention_values = torch.split(self.attn_ct_linear(only_two_entity_embed), [self.units, self.units], 2) 


        decoder_state = text_encoder_state
        decoder_output = use_cuda(torch.empty(0))
        ce_alignments = use_cuda(torch.empty(0))
        co_alignments = use_cuda(torch.empty(0))
        ct_alignments = use_cuda(torch.empty(0)) 

        # central entity mask
        local_entity_mask = np.zeros([batch_size, local_entity.shape[1]])
        for i in range(batch_size):
            local_entity_mask[i][0:local_entity_length[i]] = 1
        local_entity_mask = use_cuda(torch.from_numpy(local_entity_mask).type('torch.LongTensor'))

        # two-hop entity mask
        only_two_entity_mask = np.zeros([batch_size, only_two_entity.shape[1]])
        for i in range(batch_size):
            only_two_entity_mask[i][0:only_two_entity_length[i]] = 1
        only_two_entity_mask = use_cuda(torch.from_numpy(only_two_entity_mask).type('torch.LongTensor'))

        context = use_cuda(torch.zeros([batch_size, self.units]))
        
        if not self.is_inference:
            for t in range(decoder_len):
                decoder_input_t = torch.cat((decoder_input[:,t,:], context), 1).unsqueeze(1)
                
                decoder_output_t, decoder_state = self.decoder(decoder_input_t, decoder_state)
                context, ce_alignments_t, co_alignments_t, ct_alignments_t = self.attention(c_attention_keys, c_attention_values, \
                    ce_attention_keys, ce_attention_values, co_attention_keys, co_attention_values, ct_attention_keys, \
                    decoder_output_t.squeeze(1), local_entity_mask, only_two_entity_mask)
                decoder_output_t = context.unsqueeze(1)
                ce_alignments = torch.cat((ce_alignments, ce_alignments_t.unsqueeze(1)), 1)
                
                co_alignments = torch.cat((co_alignments, co_alignments_t.unsqueeze(1)), 1)
                decoder_output = torch.cat((decoder_output, decoder_output_t), 1)
                ct_alignments = torch.cat((ct_alignments, ct_alignments_t.unsqueeze(1)), 1)
        
        else:
            word_index = use_cuda(torch.empty(0).type('torch.LongTensor'))
            decoder_input_t = self.WordEmbedding(use_cuda(torch.ones([batch_size]).type('torch.LongTensor')))
            context = use_cuda(torch.zeros([batch_size, self.units]))
            decoder_state = text_encoder_state
            selector = use_cuda(torch.empty(0).type('torch.LongTensor'))
            
            for t in range(decoder_len):
                decoder_input_t = torch.cat((decoder_input_t, context), 1).unsqueeze(1)
                decoder_output_t, decoder_state = self.decoder(decoder_input_t, decoder_state)
                context, ce_alignments_t, co_alignments_t, ct_alignments_t = self.attention(c_attention_keys, c_attention_values, \
                    ce_attention_keys, ce_attention_values, co_attention_keys, co_attention_values, ct_attention_keys, \
                    decoder_output_t.squeeze(1), local_entity_mask, only_two_entity_mask)
                ct_alignments = torch.cat((ct_alignments, ct_alignments_t.unsqueeze(1)), 1)
                decoder_output_t = context.unsqueeze(1)
                
                decoder_input_t, word_index_t, selector_t = self.inference(decoder_output_t, ce_alignments_t, ct_alignments_t, word2id, \
                    local_entity, only_two_entity, id2entity)
                word_index = torch.cat((word_index, word_index_t.unsqueeze(1)), 1)
                selector = torch.cat((selector, selector_t.unsqueeze(1)), 1)
        
        decoder_mask = np.zeros([batch_size, decoder_len])
        for i in range(batch_size):
            decoder_mask[i][0:responses_length[i]] = 1
        decoder_mask = use_cuda(torch.from_numpy(decoder_mask).type('torch.LongTensor'))

        one_hot_entities_local = use_cuda(torch.zeros(batch_size, decoder_len, max_local_entity))
        for b in range(batch_size):
            for d in range(decoder_len):
                if match_entity_one_hop[b][d] == -1:
                    continue
                else:
                    one_hot_entities_local[b][d][match_entity_one_hop[b][d]] = 1
                
        use_entities_local = torch.sum(one_hot_entities_local, [2])
        
        one_hot_entities_only_two = use_cuda(torch.zeros(batch_size, decoder_len, max_only_two_entity))
        for b in range(batch_size):
            for d in range(decoder_len):
                if match_entity_only_two[b][d] == -1:
                    continue
                else:
                    one_hot_entities_only_two[b][d][match_entity_only_two[b][d]] = 1
                
        use_entities_only_two = torch.sum(one_hot_entities_only_two, [2])

        if not self.is_inference:
            decoder_loss, ppx_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_local, sentence_ppx_only_two, \
                word_neg_num, local_neg_num, only_two_neg_num = self.total_loss(decoder_output, responses_target, decoder_mask, \
                ce_alignments, ct_alignments, use_entities_local, one_hot_entities_local, use_entities_only_two, one_hot_entities_only_two)

        if self.is_select:
            self.sort(id2entity, ct_alignments, only_two_entity)
        
        if self.is_inference == True:
            return word_index.cpu().numpy().tolist(), selector.cpu().numpy().tolist()
        return decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_local, sentence_ppx_only_two, word_neg_num, local_neg_num, only_two_neg_num

    def sort(self, id2entity, ct_alignments, only_two_entity):
        only_two_score = torch.sum(ct_alignments, 1)
        _, sort_local_index = only_two_score.sort(1)
        sort_global_index = torch.gather(only_two_entity, 1, sort_local_index)
        sort_global_index = sort_global_index.cpu().numpy().tolist()

        sort_str = []
        for i in range(len(sort_global_index)):
            tmp = []
            for j in range(len(sort_global_index[i])):
                if sort_global_index[i][j] == 1:
                    continue    
                tmp.append(id2entity[sort_global_index[i][j]])
            sort_str.append(tmp)

        sort_f = open('selected_concept.txt','a')
        for line in sort_str:
            sort_f.write(str(line) + '\n')
        sort_f.close()
        

    def inference(self, decoder_output_t, ce_alignments_t, ct_alignments_t, word2id, local_entity, only_two_entity, id2entity):
        
        batch_size = decoder_output_t.shape[0]

        logits = self.logits_linear(decoder_output_t.squeeze(1)) # batch * num_symbols
        
        selector = self.softmax_d1(self.selector_linear(decoder_output_t.squeeze(1)))
        
        (word_prob, word_t) = torch.max(selector[:,0].unsqueeze(1) * self.softmax_d1(logits), dim = 1) 
        (local_entity_prob, local_entity_l_index_t) = torch.max(selector[:,1].unsqueeze(1) * ce_alignments_t, dim = 1)
        (only_two_entity_prob, only_two_entity_l_index_t) = torch.max(selector[:,2].unsqueeze(1) * ct_alignments_t, dim = 1)
    
        selector[:,0] = selector[:,0] * word_prob
        selector[:,1] = selector[:,1] * local_entity_prob
        selector[:,2] = selector[:,2] * only_two_entity_prob
        selector = torch.argmax(selector, dim = 1)
        
        local_entity_l_index_t = local_entity_l_index_t.cpu().numpy().tolist()
        only_two_entity_l_index_t = only_two_entity_l_index_t.cpu().numpy().tolist()
        word_t = word_t.cpu().numpy().tolist()

        word_local_entity_t = []
        word_only_two_entity_t = []
        word_index_final_t = []
        for i in range(batch_size):
            if selector[i] == 0:
                word_index_final_t.append(word_t[i])
                continue
            if selector[i] == 1:
                local_entity_index_t = int(local_entity[i][local_entity_l_index_t[i]])
                local_entity_text = id2entity[local_entity_index_t]
                if local_entity_text not in word2id:
                    local_entity_text = '_UNK'
                word_index_final_t.append(word2id[local_entity_text])
                continue
            if selector[i] == 2:
                only_two_entity_index_t = int(only_two_entity[i][only_two_entity_l_index_t[i]])
                only_two_entity_text = id2entity[only_two_entity_index_t]
                if only_two_entity_text not in word2id:
                    only_two_entity_text = '_UNK'
                word_index_final_t.append(word2id[only_two_entity_text])
                continue

        word_index_final_t = use_cuda(torch.LongTensor(word_index_final_t))
        decoder_input_t = self.WordEmbedding(word_index_final_t)

        return decoder_input_t, word_index_final_t, selector

    def total_loss(self, decoder_output, responses_target, decoder_mask, ce_alignments, ct_alignments, use_entities_local, \
        entity_targets_local, use_entities_only_two, entity_targets_only_two):
        batch_size = decoder_output.shape[0]
        decoder_len = responses_target.shape[1]
        
        local_masks = use_cuda(decoder_mask.reshape([-1]).type("torch.FloatTensor"))
        local_masks_word = use_cuda((1 - use_entities_local - use_entities_only_two).reshape([-1]).type("torch.FloatTensor")) * local_masks
        local_masks_local = use_cuda(use_entities_local.reshape([-1]).type("torch.FloatTensor"))
        local_masks_only_two = use_cuda(use_entities_only_two.reshape([-1]).type("torch.FloatTensor"))
        logits = self.logits_linear(decoder_output) #batch * decoder_len * num_symbols
        
        word_prob = torch.gather(self.softmax_d2(logits), 2, responses_target.unsqueeze(2)).squeeze(2)
        
        selector_word, selector_local, selector_only_two = torch.split(self.softmax_d2(self.selector_linear(decoder_output)), [1, 1, 1], 2) #batch_size * decoder_len * 1
        selector_word = selector_word.squeeze(2)
        selector_local = selector_local.squeeze(2)
        selector_only_two = selector_only_two.squeeze(2)

        entity_prob_local = torch.sum(ce_alignments * entity_targets_local, [2])
        entity_prob_only_two = torch.sum(ct_alignments * entity_targets_only_two, [2])

        ppx_prob = word_prob * (1 - use_entities_local - use_entities_only_two) + entity_prob_local * use_entities_local + entity_prob_only_two * use_entities_only_two
        ppx_word = word_prob * (1 - use_entities_local - use_entities_only_two)
        ppx_local = entity_prob_local * use_entities_local
        ppx_only_two = entity_prob_only_two * use_entities_only_two

        final_prob = word_prob * selector_word * (1 - use_entities_local - use_entities_only_two) + entity_prob_local * selector_local * \
            use_entities_local + entity_prob_only_two * selector_only_two * use_entities_only_two
        
        final_loss = torch.sum(- torch.log(1e-12 + final_prob).reshape([-1]) * local_masks)

        sentence_ppx = torch.sum((- torch.log(1e-12 + ppx_prob).reshape([-1]) * local_masks).reshape([batch_size, -1]), 1)
        sentence_ppx_word = torch.sum((- torch.log(1e-12 + ppx_word).reshape([-1]) * local_masks_word).reshape([batch_size, -1]), 1)
        sentence_ppx_local = torch.sum((- torch.log(1e-12 + ppx_local).reshape([-1]) * local_masks_local).reshape([batch_size, -1]), 1)
        sentence_ppx_only_two = torch.sum((- torch.log(1e-12 + ppx_only_two).reshape([-1]) * local_masks_only_two).reshape([batch_size, -1]), 1)
        
        selector_loss = torch.sum(- torch.log(1e-12 + selector_local * use_entities_local + selector_only_two * use_entities_only_two + \
            selector_word * (1 - use_entities_local - use_entities_only_two)).reshape([-1]) * local_masks)
        
        loss = final_loss + selector_loss
        total_size = torch.sum(local_masks)
        total_size += 1e-12 

        sum_word = torch.sum(use_cuda(((1 - use_entities_local - use_entities_only_two) * use_cuda(decoder_mask.type("torch.FloatTensor"))).type("torch.FloatTensor")), 1)
        sum_local = torch.sum(use_cuda(use_entities_local.type("torch.FloatTensor")), 1)
        sum_only_two= torch.sum(use_cuda(use_entities_only_two.type("torch.FloatTensor")), 1)

        word_neg_mask = use_cuda((sum_word == 0).type("torch.FloatTensor"))
        local_neg_mask = use_cuda((sum_local == 0).type("torch.FloatTensor"))
        only_two_neg_mask = use_cuda((sum_only_two == 0).type("torch.FloatTensor"))

        word_neg_num = torch.sum(word_neg_mask)
        local_neg_num = torch.sum(local_neg_mask)
        only_two_neg_num = torch.sum(only_two_neg_mask)

        sum_word = sum_word + word_neg_mask
        sum_local = sum_local + local_neg_mask
        sum_only_two = sum_only_two + only_two_neg_mask

        return loss / total_size, 0, sentence_ppx / torch.sum(use_cuda(decoder_mask.type("torch.FloatTensor")), 1), \
            sentence_ppx_word / sum_word, sentence_ppx_local / sum_local, sentence_ppx_only_two / sum_only_two, word_neg_num, \
            local_neg_num, only_two_neg_num
        
        

    def attention(self, c_attention_keys, c_attention_values, ce_attention_keys, ce_attention_values, co_attention_keys, \
        co_attention_values, ct_attention_keys, decoder_state, local_entity_mask, only_two_entity_mask):
        batch_size = ct_attention_keys.shape[0]
        only_two_len = ct_attention_keys.shape[1]

        c_query = decoder_state.reshape([-1, 1, self.units])
        ce_query = decoder_state.reshape([-1, 1, self.units])
        co_query = decoder_state.reshape([-1, 1, self.units])
        ct_query = decoder_state.reshape([-1, 1, self.units])
        
        c_scores = torch.sum(c_attention_keys * c_query, 2)
        ce_scores = torch.sum(ce_attention_keys * ce_query, 2)
        co_scores = torch.sum(co_attention_keys * co_query, 2)
        ct_scores = torch.sum(ct_attention_keys * ct_query, 2)

        c_alignments = self.softmax_d1(c_scores)
        ce_alignments = self.softmax_d1(ce_scores)
        co_alignments = self.softmax_d1(co_scores)
        ct_alignments = self.softmax_d1(ct_scores)

        ce_alignments = ce_alignments * use_cuda(local_entity_mask.type("torch.FloatTensor"))
        ct_alignments = ct_alignments * use_cuda(only_two_entity_mask.type("torch.FloatTensor"))
        
        c_context = torch.sum(c_alignments.unsqueeze(2) * c_attention_values, 1)
        ce_context = torch.sum(ce_alignments.unsqueeze(2) * ce_attention_values, 1)
        co_context = torch.sum(co_alignments.unsqueeze(2) * co_attention_values, 1)

        context = self.context_linear(torch.cat((decoder_state, c_context, ce_context, co_context), 1))
             
        return context, ce_alignments, co_alignments, ct_alignments
