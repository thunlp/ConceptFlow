# ConceptFlow

This is the implementation of ConceptFlow described in ACL 2020 paper [Grounded Conversation Generation as Guided Traverses in Commonsense Knowledge Graphs](https://www.aclweb.org/anthology/2020.acl-main.184.pdf).

### Prerequisites
The recommended way to install the required packages is using pip and the provided `requirements.txt` file. Create the environment by running the following command:
```
pip install -r requirements.txt
```

### Download Dataset
* Due to the policy of Reddit, we are not able to release the data in a public repo. Please send email to ```hzhan148@cs.brown.edu``` to request data.
* By default, we expect the data to be stored in `./data`.


### Train and inference

For training, edit `config.yml` and set `is_train: True`. Run `python train.py`. Training result will be output to `./training_output`.

For inference, edit `config.yml`, set `is_train: False` and `test_model_path: 'Your Model Path'`. Run `python inference.py`. Generated responses will be output to `./inference_output`.

### Concept Selection

For concept selection, edit `config.yml` set `is_train: False`, `test_model_path: 'Your Selector Path'` and `is_select: True`. Run `python sort.py`. The sorted two-hop concepts will be output to `selected_concept.txt` with ascending order.

### Evaluation

To evaluate the generated response, we use the metrics and the scripts of [DSTC7](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling/tree/master/evaluation). Also, we use this [implementation](https://github.com/pltrdy/rouge) to calculate ROUGE.

### Overall Results
* Relevance Between Generated and Golden Responses. The PPL results of GPT-2 is not directly comparable because of its different tokenization.

| Model | Bleu-4 | Nist-4 | Rouge-1 | Rouge-2 | Rouge-L | Meteor | PPL |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Seq2seq | 0.0098 | 1.1069 | 0.1441 | 0.0189 | 0.1146 | 0.0611 | 48.79 |
| MemNet | 0.0112 | 1.1977 | 0.1523 | 0.0215 | 0.1213 | 0.0632 | 47.38 |
| CopyNet | 0.0106 | 1.0788 | 0.1472 | 0.0211 | 0.1153 | 0.0610 | 43.28 |
| CCM | 0.0084 | 0.9095 | 0.1538| 0.0211 | 0.1245 | 0.0630 | 42.91 |
| GPT-2 (lang) | 0.0162 | 1.0844 | 0.1321 | 0.0117 | 0.1046  | 0.0637 | 29.08 |
| GPT-2 (conv) | 0.0124 | 1.1763 | 0.1514 | 0.0222 | 0.1212 | 0.0629 | 24.55 |
| ConceptFlow | 0.0246 | 1.8329 | 0.2280 | 0.0469 | 0.1888 | 0.0942 | 29.90 |

*  Diversity of Generated Response.

| Model | Dist-1 | Dist-2 | Ent-4 |
| --- | --- | --- | --- |
| Seq2seq | 0.0123 | 0.0525 | 7.665 |
| MemNet | 0.0211 | 0.0931 | 8.418 |
| CopyNet | 0.0223 | 0.0988 | 8.422 |
| CCM | 0.0146 | 0.0643 | 7.847 |
| GPT-2 (lang) | 0.0325 | 0.2461 | 11.65 |
| GPT-2 (conv) | 0.0266 | 0.1218 | 8.546 |
| ConceptFlow | 0.0223 | 0.1228 | 10.27 |



### Citation

```
@inproceedings{zhang-etal-2020-grounded,
    title = "Grounded Conversation Generation as Guided Traverses in Commonsense Knowledge Graphs",
    author = "Zhang, Houyu  and
      Liu, Zhenghao  and
      Xiong, Chenyan  and
      Liu, Zhiyuan",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.acl-main.184",
    pages = "2031--2043",
    abstract = "Human conversations naturally evolve around related concepts and hop to distant concepts. This paper presents a new conversation generation model, ConceptFlow, which leverages commonsense knowledge graphs to explicitly model conversation flows. By grounding conversations to the concept space, ConceptFlow represents the potential conversation flow as traverses in the concept space along commonsense relations. The traverse is guided by graph attentions in the concept graph, moving towards more meaningful directions in the concept space, in order to generate more semantic and informative responses. Experiments on Reddit conversations demonstrate ConceptFlow{'}s effectiveness over previous knowledge-aware conversation models and GPT-2 based models while using 70{\%} fewer parameters, confirming the advantage of explicit modeling conversation structures. All source codes of this work are available at https://github.com/thunlp/ConceptFlow.",
}
```

### Aceknowledgements
This code was based in part on the source code of [CCM](https://github.com/tuxchow/ccm) and [GraftNet](https://github.com/OceanskySun/GraftNet).

### Contact
If you have any question or suggestion, please send email to:

```hzhan148@cs.brown.edu```
