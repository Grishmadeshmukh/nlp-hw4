import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        # TODO
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.x = load_lines(os.path.join(data_folder, f"{split}.nl"))
        if split != "test":
            self.y = load_lines(os.path.join(data_folder, f"{split}.sql"))
        else:
            self.y = None

    def process_data(self, data_folder, split, tokenizer):
        # TODO
        pass
    
    def __len__(self):
        # TODO
        return len(self.x)

    def __getitem__(self, idx):
        # TODO
        inp = "translate English to SQL: " + self.x[idx]
        enc = self.tokenizer(inp, return_tensors="pt", truncation=True)
        enc_ids = enc.input_ids.squeeze(0)
        if self.split == "test":
            return enc_ids
        target = self.tokenizer(self.y[idx], return_tensors="pt", truncation=True).input_ids.squeeze(0)
        decoder_input = target[:-1]
        decoder_target = target[1:]
        return enc_ids, decoder_input, decoder_target

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    # return [], [], [], [], []
    enc = [b[0] for b in batch]
    dec_in = [b[1] for b in batch]
    dec_tg = [b[2] for b in batch]
    enc = pad_sequence(enc, batch_first=True, padding_value=PAD_IDX)
    dec_in = pad_sequence(dec_in, batch_first=True, padding_value=PAD_IDX)
    dec_tg = pad_sequence(dec_tg, batch_first=True, padding_value=PAD_IDX)
    mask = (enc != PAD_IDX).long()
    init = dec_in[:, :1]
    return enc, mask, dec_in, dec_tg, init


def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # TODO
    # return [], [], []
    enc = pad_sequence(batch, batch_first=True, padding_value=PAD_IDX)
    mask = (enc != PAD_IDX).long()
    init = torch.ones((enc.size(0),1), dtype=torch.long)
    return enc, mask, init, init, init

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x