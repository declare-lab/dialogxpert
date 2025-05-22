from prompt import ESConvAct, CIMAAct, CBAct
from qwen_prompts import *
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import torch
import os
import pickle
import json

class DataReader(Dataset):

    def __init__(self, data, args):

        self._source_ids = data['source_ids']
        self._target_ids = data['target_ids']
        self._max_len = args.max_seq_length

    def __len__(self):
        return len(self._source_ids)
    
    def __getitem__(self, index):
        return self._source_ids[index][:self._max_len], self._target_ids[index]

def _collate_fn(batch):

    source_ids, target_ids = zip(*batch)

    # Convert to tensors
    input_ids = [torch.LongTensor(source_id) for source_id in source_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': input_ids.ne(0),
        'labels': torch.LongTensor(target_ids)
    }

def get_action_list(dataset):
    return {
        'esc': ESConvAct, 
        'cima': CIMAAct, 
        'cb': CBAct,
        'extes': ExTESAct,
        'p4g': P4G_Act
    }[dataset]

def convert_dataloader(dataset, sampler, args):

    return DataLoader(
        DataReader(dataset, args), 
        batch_size=args.train_batch_size, shuffle=True, 
        collate_fn=_collate_fn
    )
    
