# ConceptFlow

This is the implementation of ConceptFlow described in ACL 2020 paper [Grounded Conversation Generation as Guided Traverses in Commonsense Knowledge Graphs]().

### Prerequisites
The recommended way to install the required packages is using pip and the provided `requirements.yml` file. Create the environment by running the following command:
```
pip install -r requirements.txt
```

### Download Dataset
* [Dataset](https://drive.google.com/open?id=1wjcjY20jMyktHcQ_-ez0vyu9lh2gik1F)

Download the above data and put them into the `./data` folder.

### Train and Test

To train model, edit Config class in `main.py`, set `self.is_train = True`. 

To test model, edit Config class in `main.py`, set `self.is_train = False` and `self.test_model_path = 'Your Model Path'. 

After the setting of Config class, all left is to run the following command:

```
python main.py
```

### Evaluation

To evaluate the generated result, we use the metrics and the scripts of DSTC7. Click [here](https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling/tree/master/evaluation) for details. Also, we use this [implementation](https://github.com/pltrdy/rouge) to calculate ROUGE.

### Aceknowledgements
This code was based in part on the source code of [CCM](https://github.com/tuxchow/ccm) and [GraftNet](https://github.com/OceanskySun/GraftNet).
