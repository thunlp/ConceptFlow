#coding:utf-8
import numpy as np
import json
from model import ConceptFlow, use_cuda
from preprocession import prepare_data, build_vocab, gen_batched_data
import torch
import warnings
import yaml
import os
warnings.filterwarnings('ignore')

csk_triples, csk_entities, kb_dict = [], [], []
dict_csk_entities, dict_csk_triples = {}, {}
class Config():
    def __init__(self, path):
        self.config_path = path
        self._get_config()

    def _get_config(self):
        with open(self.config_path, "r") as setting:
            config = yaml.load(setting)
        self.is_train = config['is_train']
        self.is_select = config['is_select']
        self.test_model_path = config['test_model_path']
        self.embed_units = config['embed_units']
        self.symbols = config['symbols']
        self.units = config['units']
        self.layers = config['layers']
        self.batch_size = config['batch_size']
        self.data_dir = config['data_dir']
        self.num_epoch = config['num_epoch']
        self.lr_rate = config['lr_rate']
        self.lstm_dropout = config['lstm_dropout']
        self.linear_dropout = config['linear_dropout']
        self.max_gradient_norm = config['max_gradient_norm']
        self.trans_units = config['trans_units']
        self.gnn_layers = config['gnn_layers']
        self.fact_dropout = config['fact_dropout']
        self.fact_scale = config['fact_scale']
        self.pagerank_lambda = config['pagerank_lambda']
        self.result_dir_name = config['result_dir_name']
        self.generated_path = config['generated_path']

    def list_all_member(self):
        for name, value in vars(self).items():
            print('%s = %s' % (name, value))
        

def run(model, data_train, config, word2id, entity2id):
    batched_data = gen_batched_data(data_train, config, word2id, entity2id)
    
    if model.is_inference == True:
        word_index, selector = model(batched_data)
        return word_index, selector
    else:
        decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_local, sentence_ppx_only_two, word_neg_num, local_neg_num, only_two_neg_num = model(batched_data)
        return decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_local, sentence_ppx_only_two, word_neg_num, local_neg_num, only_two_neg_num


def sort(model, data_test, config, word2id, entity2id, model_path=None):
    if model_path != None:
        model.load_state_dict(torch.load(model_path))

    count = 0
    model.is_inference = True

    for iteration in range(len(data_test) // config.batch_size):
        
        _, _ = run(model, data_test[(iteration * config.batch_size):(iteration * \
            config.batch_size + config.batch_size)], config, word2id, entity2id)

        if count % 50 == 0:
            print ("sort:", iteration)
        count += 1 

def main():
    config = Config('config.yml')
    config.list_all_member()
    raw_vocab, _, data_test = prepare_data(config)
    word2id, entity2id, vocab, embed, entity_vocab, entity_embed, relation_vocab, relation_embed, entity_relation_embed = build_vocab(config.data_dir, raw_vocab, config = config)  
    model = use_cuda(ConceptFlow(config, embed, entity_relation_embed, is_select=config.is_select))

    model_optimizer = torch.optim.Adam(model.parameters(), lr = config.lr_rate)   
    
    if not os.path.exists(config.generated_path):
        os.mkdir(config.generated_path)

    sort(model, data_test, config, word2id, entity2id, model_path=config.test_model_path)

main()
