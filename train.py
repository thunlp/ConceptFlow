#coding:utf-8
import numpy as np
import json
from model import ConceptFlow, use_cuda
from preprocession import prepare_data, build_vocab, get_data
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

    def list_all_member(self):
        for name, value in vars(self).items():
            print('%s = %s' % (name, value))
        

def run(model, batched_data):
    if model.is_inference == True:
        word_index, selector = model(batched_data)
        return word_index, selector
    else:
        decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_local, sentence_ppx_only_two, word_neg_num, local_neg_num, only_two_neg_num = model(batched_data)
        return decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_local, sentence_ppx_only_two, word_neg_num, local_neg_num, only_two_neg_num

def train(config, model, data_train, data_test, word2id, entity2id, model_optimizer):
    for epoch in range(config.num_epoch):
            print ("epoch: ", epoch)
            sentence_ppx_loss = 0
            sentence_ppx_word_loss = 0
            sentence_ppx_local_loss = 0
            sentence_ppx_only_two_loss = 0

            word_cut = use_cuda(torch.Tensor([0]))
            local_cut = use_cuda(torch.Tensor([0]))
            only_two_cut = use_cuda(torch.Tensor([0]))

            count = 0
            for iteration, batch in enumerate(data_train):
                decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_local, sentence_ppx_only_two, word_neg_num, local_neg_num, \
                    only_two_neg_num = run(model, batch, config, word2id, entity2id)
                sentence_ppx_loss += torch.sum(sentence_ppx).data
                sentence_ppx_word_loss += torch.sum(sentence_ppx_word).data
                sentence_ppx_local_loss += torch.sum(sentence_ppx_local).data
                sentence_ppx_only_two_loss += torch.sum(sentence_ppx_only_two).data

                word_cut += word_neg_num
                local_cut += local_neg_num
                only_two_cut += only_two_neg_num

                model_optimizer.zero_grad()
                decoder_loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), config.max_gradient_norm)
                model_optimizer.step()
                
                if count % 50 == 0:
                    print ("iteration:", iteration, "Loss:", decoder_loss.data)
                count += 1
            
            print ("perplexity for epoch", epoch + 1, ":", np.exp(sentence_ppx_loss.cpu() / len(data_train)), " ppx_word: ", \
                np.exp(sentence_ppx_word_loss.cpu() / (len(data_train) - int(word_cut))), " ppx_local: ", \
                np.exp(sentence_ppx_local_loss.cpu() / (len(data_train) - int(local_cut))), " ppx_only_two: ", \
                np.exp(sentence_ppx_only_two_loss.cpu() / (len(data_train) - int(only_two_cut))))

            torch.save(model.state_dict(), config.result_dir_name + '/' + '_epoch_' + str(epoch + 1) + '.pkl')
            ppx, ppx_word, ppx_local, ppx_only_two = evaluate(model, data_test, config, word2id, entity2id, epoch + 1)
            ppx_f = open(config.result_dir_name + '/result.txt','a')
            ppx_f.write("epoch " + str(epoch + 1) + " ppx: " + str(ppx) + " ppx_word: " + str(ppx_word) + " ppx_local: " + \
                str(ppx_local) + " ppx_only_two: " + str(ppx_only_two) + '\n')
            ppx_f.close()

def evaluate(model, data_test, config, word2id, entity2id, epoch = 0, model_path = None):
    if model_path != None:
        model.load_state_dict(torch.load(model_path))
    sentence_ppx_loss = 0
    sentence_ppx_word_loss = 0
    sentence_ppx_local_loss = 0
    sentence_ppx_only_two_loss = 0
    word_cut = use_cuda(torch.Tensor([0]))
    local_cut = use_cuda(torch.Tensor([0]))
    only_two_cut = use_cuda(torch.Tensor([0]))
    count = 0
    id2word = dict()
    for key in word2id.keys():
        id2word[word2id[key]] = key


    for iteration, batch in enumerate(data_test):
        
        decoder_loss, sentence_ppx, sentence_ppx_word, sentence_ppx_local, sentence_ppx_only_two, word_neg_num, \
            local_neg_num, only_two_neg_num = run(model, batch, config, word2id, entity2id)
        sentence_ppx_loss += torch.sum(sentence_ppx).data
        sentence_ppx_word_loss += torch.sum(sentence_ppx_word).data
        sentence_ppx_local_loss += torch.sum(sentence_ppx_local).data
        sentence_ppx_only_two_loss += torch.sum(sentence_ppx_only_two).data

        word_cut += word_neg_num
        local_cut += local_neg_num
        only_two_cut += only_two_neg_num

        if count % 50 == 0:
            print ("iteration for evaluate:", iteration, "Loss:", decoder_loss.data)
        count += 1
        
    model.is_inference = False
    if model_path != None:
        print('    perplexity on test set:', np.exp(sentence_ppx_loss.cpu() / len(data_test)), \
            np.exp(sentence_ppx_word_loss.cpu() / (len(data_test) - int(word_cut))), np.exp(sentence_ppx_local_loss.cpu() / (len(data_test) \
            - int(local_cut))), np.exp(sentence_ppx_only_two_loss.cpu() / (len(data_test) - int(only_two_cut))))
        exit()
    print('    perplexity on test set:', np.exp(sentence_ppx_loss.cpu() / len(data_test)), np.exp(sentence_ppx_word_loss.cpu() / \
        (len(data_test) - int(word_cut))), np.exp(sentence_ppx_local_loss.cpu() / (len(data_test) - int(local_cut))), \
        np.exp(sentence_ppx_only_two_loss.cpu() / (len(data_test) - int(only_two_cut))))
    return np.exp(sentence_ppx_loss.cpu() / len(data_test)), np.exp(sentence_ppx_word_loss.cpu() / (len(data_test) - int(word_cut))), \
        np.exp(sentence_ppx_local_loss.cpu() / (len(data_test) - int(local_cut))), np.exp(sentence_ppx_only_two_loss.cpu() / \
        (len(data_test) - int(only_two_cut)))

def main():
    config = Config('config.yml')
    config.list_all_member()
    raw_vocab = prepare_data(config)
    word2id, entity2id, vocab, embed, entity_vocab, entity_embed, relation_vocab, relation_embed, entity_relation_embed = build_vocab(config.data_dir, raw_vocab, config = config)  
    data_train, data_test = get_data(config,word2id,entity2id)

    model = use_cuda(ConceptFlow(config, embed, entity_relation_embed))

    model_optimizer = torch.optim.Adam(model.parameters(), lr = config.lr_rate)   
    
    if not os.path.exists(config.result_dir_name):
        os.mkdir(config.result_dir_name)
    ppx_f = open(config.result_dir_name + '/result.txt','a')
    for name, value in vars(config).items():
        ppx_f.write('%s = %s' % (name, value) + '\n')

    if config.is_train == False:
        evaluate(model, data_test, config, word2id, entity2id, 0, model_path = config.test_model_path)
        exit() 
    train(config, model, data_train, data_test, word2id, entity2id, model_optimizer)

main()
