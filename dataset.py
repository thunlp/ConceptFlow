import json
from torch.utils.data import IterableDataset

class ConceptFlowDataset(IterableDataset):
    def __init__(self, txt_file, config):
        self.root_dir = config.data_dir        
        self.txt_file = txt_file

    def __iter__(self):
        f = open(f'{self.root_dir}/{self.txt_file}')
        return map(json.loads, f)