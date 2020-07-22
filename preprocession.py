#coding:utf-8
import numpy as np
import json
import torch
from utils import padding, padding_triple_id, build_kb_adj_mat

        
def prepare_data(config):
    global csk_entities, csk_triples, kb_dict, dict_csk_entities, dict_csk_triples
    
    with open('%s/resource.txt' % config.data_dir) as f:
        d = json.loads(f.readline())
    
    csk_triples = d['csk_triples']
    csk_entities = d['csk_entities']
    raw_vocab = d['vocab_dict']
    kb_dict = d['dict_csk']
    dict_csk_entities = d['dict_csk_entities']
    dict_csk_triples = d['dict_csk_triples']
    
    data_train, data_test = [], []

    if config.is_train:
        with open('%s/trainset.txt' % config.data_dir) as f:
            for idx, line in enumerate(f):
                if idx % 100000 == 0: print('read train file line %d' % idx)
                data_train.append(json.loads(line))

    
    with open('%s/testset.txt' % config.data_dir) as f:
        for line in f:
            data_test.append(json.loads(line))
    
    return raw_vocab, data_train, data_test

def build_vocab(path, raw_vocab, config, trans='transE'):

    print("Creating word vocabulary...")
    vocab_list = ['_PAD','_GO', '_EOS', '_UNK', ] + sorted(raw_vocab, key=raw_vocab.get, reverse=True)
    if len(vocab_list) > config.symbols:
        vocab_list = vocab_list[:config.symbols]
    
    print("Creating entity vocabulary...")
    entity_list = ['_NONE', '_PAD_H', '_PAD_R', '_PAD_T', '_NAF_H', '_NAF_R', '_NAF_T'] 
    with open('%s/entity.txt' % path) as f:
        for i, line in enumerate(f):
            e = line.strip()
            entity_list.append(e)
    
    print("Creating relation vocabulary...")
    relation_list = []
    with open('%s/relation.txt' % path) as f:
        for i, line in enumerate(f):
            r = line.strip()
            relation_list.append(r)

    print("Loading word vectors...")
    vectors = {}
    with open('%s/glove.840B.300d.txt' % path) as f:
        for i, line in enumerate(f):
            if i % 100000 == 0:
                print("    processing line %d" % i)
            s = line.strip()
            word = s[:s.find(' ')]
            vector = s[s.find(' ')+1:]
            vectors[word] = vector
    
    embed = []
    for word in vocab_list:
        if word in vectors:
            #vector = map(float, vectors[word].split())
            vector = vectors[word].split()
        else:
            vector = np.zeros((config.embed_units), dtype=np.float32) 
        embed.append(vector)
    embed = np.array(embed, dtype=np.float32)
            
    print("Loading entity vectors...")
    entity_embed = []
    with open('%s/entity_%s.txt' % (path, trans)) as f:
        for i, line in enumerate(f):
            s = line.strip().split('\t')
            #entity_embed.append(map(float, s))
            entity_embed.append(s)

    print("Loading relation vectors...")
    relation_embed = []
    with open('%s/relation_%s.txt' % (path, trans)) as f:
        for i, line in enumerate(f):
            s = line.strip().split('\t')
            relation_embed.append(s)

    entity_relation_embed = np.array(entity_embed+relation_embed, dtype=np.float32)
    entity_embed = np.array(entity_embed, dtype=np.float32)
    relation_embed = np.array(relation_embed, dtype=np.float32)

    word2id = dict()
    entity2id = dict()
    for word in vocab_list:
        word2id[word] = len(word2id)
    for entity in entity_list + relation_list:
        entity2id[entity] = len(entity2id)

    return word2id, entity2id, vocab_list, embed, entity_list, entity_embed, relation_list, relation_embed, entity_relation_embed

def gen_batched_data(data, config, word2id, entity2id): 
    global csk_entities, csk_triples, kb_dict, dict_csk_entities, dict_csk_triples

    encoder_len = max([len(item['post']) for item in data])+1  

    decoder_len = max([len(item['response']) for item in data])+1
    triple_num = max([len(item['all_triples_one_hop']) for item in data])
    entity_len = max([len(item['all_entities_one_hop']) + max(item['post_triples']) for item in data])
    only_two_entity_len = max([len(item['only_two']) for item in data])
    triple_num_one_two = max([len(item['one_two_triple']) for item in data]) 
    triple_len_one_two = max([len(tri) for item in data for tri in item['one_two_triple']]) 
    posts_id = np.full((len(data), encoder_len), 0, dtype=int)
    responses_id = np.full((len(data), decoder_len), 0, dtype=int)
    responses_length = []
    # posts_length = []
    local_entity_length = []
    only_two_entity_length = []
    local_entity = []
    only_two_entity = []
    kb_fact_rels = np.full((len(data), triple_num), 2, dtype=int)
    kb_adj_mats = np.empty(len(data), dtype=object)
    q2e_adj_mats = np.full((len(data), entity_len), 0, dtype=int)
    match_entity_one_hop = np.full((len(data), decoder_len), -1, dtype=int)
    match_entity_only_two = np.full((len(data), decoder_len), -1, dtype=int)
    one_two_triples_id = []
    g2l_only_two_list = []
    # o2t_entity_index_list = []

    next_id = 0
    for item in data:
        # posts
        for i, post_word in enumerate(padding(item['post'], encoder_len)):
            if post_word in word2id:
                posts_id[next_id, i] = word2id[post_word]
                
            else:
                posts_id[next_id, i] = word2id['_UNK']

        # responses
        for i, response_word in enumerate(padding(item['response'], decoder_len)):
            if response_word in word2id:
                responses_id[next_id, i] = word2id[response_word]
                
            else:
                responses_id[next_id, i] = word2id['_UNK']

        # responses_length
        responses_length.append(len(item['response']) + 1)

        # local_entity
        local_entity_tmp = []
        for i in range(len(item['post_triples'])):
            if item['post_triples'][i] == 0:
                continue
            elif item['post'][i] not in entity2id:
                continue
            elif entity2id[item['post'][i]] in local_entity_tmp:
                continue
            else:
                local_entity_tmp.append(entity2id[item['post'][i]])

        for entity_index in item['all_entities_one_hop']:
            if csk_entities[entity_index] not in entity2id:
                continue
            if entity2id[csk_entities[entity_index]] in local_entity_tmp:
                continue
            else:
                local_entity_tmp.append(entity2id[csk_entities[entity_index]])
        local_entity_len_tmp = len(local_entity_tmp)
        local_entity_tmp += [1] * (entity_len - len(local_entity_tmp))
        local_entity.append(local_entity_tmp)

        # kb_adj_mat and kb_fact_rel
        g2l = dict()
        for i in range(len(local_entity_tmp)):
            g2l[local_entity_tmp[i]] = i

        entity2fact_e, entity2fact_f = [], []
        fact2entity_f, fact2entity_e = [], []

        tmp_count = 0
        for i in range(len(item['all_triples_one_hop'])):
            sbj = csk_triples[item['all_triples_one_hop'][i]].split()[0][:-1]
            rel = csk_triples[item['all_triples_one_hop'][i]].split()[1][:-1]
            obj = csk_triples[item['all_triples_one_hop'][i]].split()[2]

            if (sbj not in entity2id) or (obj not in entity2id):
                continue
            if (entity2id[sbj] not in g2l) or (entity2id[obj] not in g2l):
                continue
            
            entity2fact_e += [g2l[entity2id[sbj]]] 
            entity2fact_f += [tmp_count] 
            fact2entity_f += [tmp_count]
            fact2entity_e += [g2l[entity2id[obj]]]
            kb_fact_rels[next_id, tmp_count] = entity2id[rel] 
            tmp_count += 1

        kb_adj_mats[next_id] = (np.array(entity2fact_f, dtype=int), np.array(entity2fact_e, dtype=int), np.array([1.0] * len(entity2fact_f))), (np.array(fact2entity_e, dtype=int), np.array(fact2entity_f, dtype=int), np.array([1.0] * len(fact2entity_e)))
        
        # q2e_adj_mat
        for i in range(len(item['post_triples'])):
            if item['post_triples'][i] == 0:
                continue
            elif item['post'][i] not in entity2id:
                continue
            else:
                q2e_adj_mats[next_id, g2l[entity2id[item['post'][i]]]] = 1

        # match_entity_one_hop
        for i in range(len(item['match_response_index_one_hop'])):
            if item['match_response_index_one_hop'][i] == -1:
                continue
            if csk_entities[item['match_response_index_one_hop'][i]] not in entity2id:
                continue
            if entity2id[csk_entities[item['match_response_index_one_hop'][i]]] not in g2l:
                continue
            else:
                match_entity_one_hop[next_id, i] = g2l[entity2id[csk_entities[item['match_response_index_one_hop'][i]]]]

        # only_two_entity
        only_two_entity_tmp = []
        for entity_index in item['only_two']:
            if csk_entities[entity_index] not in entity2id:
                continue
            if entity2id[csk_entities[entity_index]] in only_two_entity_tmp:
                continue
            else:
                only_two_entity_tmp.append(entity2id[csk_entities[entity_index]])
        only_two_entity_len_tmp = len(only_two_entity_tmp)
        only_two_entity_tmp += [1] * (only_two_entity_len - len(only_two_entity_tmp))
        only_two_entity.append(only_two_entity_tmp)

        # match_entity_two_hop
        g2l_only_two = dict()
        for i in range(len(only_two_entity_tmp)):
            g2l_only_two[only_two_entity_tmp[i]] = i

        for i in range(len(item['match_response_index_only_two'])):
            if item['match_response_index_only_two'][i] == -1:
                continue
            if csk_entities[item['match_response_index_only_two'][i]] not in entity2id:
                continue
            else:
                match_entity_only_two[next_id, i] = g2l_only_two[entity2id[csk_entities[item['match_response_index_only_two'][i]]]]
        
        # one_two_triple
        one_two_triples_id.append(padding_triple_id(entity2id, [[csk_triples[x].split(', ') for x in triple] for triple in item['one_two_triple']], triple_num_one_two, triple_len_one_two))
        
        ############################ g2l_only_two
        g2l_only_two_list.append(g2l_only_two)

        # local_entity_length
        local_entity_length.append(local_entity_len_tmp)

        # only_two_entity_length
        only_two_entity_length.append(only_two_entity_len_tmp)

        next_id += 1

    batched_data = {'query_text': np.array(posts_id), 
            'answer_text': np.array(responses_id), 
            'local_entity': np.array(local_entity),    
            'responses_length': responses_length, 
            'q2e_adj_mat': np.array(q2e_adj_mats),
            'kb_adj_mat': build_kb_adj_mat(kb_adj_mats, config.fact_dropout),
            'kb_fact_rel': np.array(kb_fact_rels),
            'match_entity_one_hop': np.array(match_entity_one_hop),
            'only_two_entity': np.array(only_two_entity),
            'match_entity_only_two': np.array(match_entity_only_two),
            'one_two_triples_id': np.array(one_two_triples_id),
            'word2id': word2id,
            'entity2id': entity2id,
            'local_entity_length': local_entity_length,
            'only_two_entity_length': only_two_entity_length}
    
    return batched_data
