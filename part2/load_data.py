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
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
        self.pad_token_id = self.tokenizer.pad_token_id
        # T5 uses pad token as decoder start token.
        self.decoder_start_token_id = self.tokenizer.pad_token_id

        data = self.process_data(data_folder, split, self.tokenizer)
        self.encoder_ids = data["encoder_ids"]
        self.encoder_mask = data["encoder_mask"]
        self.decoder_ids = data.get("decoder_ids")
        self.initial_decoder_ids = data["initial_decoder_ids"]

    def process_data(self, data_folder, split, tokenizer):
        nl_path = os.path.join(data_folder, f"{split}.nl")
        nl_lines = load_lines(nl_path)

        encoder_out = tokenizer(
            nl_lines,
            truncation=True,
            padding=False,
            return_attention_mask=True,
        )

        initial_decoder_ids = [[self.decoder_start_token_id] for _ in nl_lines]
        processed = {
            "encoder_ids": encoder_out["input_ids"],
            "encoder_mask": encoder_out["attention_mask"],
            "initial_decoder_ids": initial_decoder_ids,
        }

        if split != "test":
            sql_path = os.path.join(data_folder, f"{split}.sql")
            sql_lines = load_lines(sql_path)
            decoder_out = tokenizer(sql_lines, truncation=True, padding=False)
            processed["decoder_ids"] = decoder_out["input_ids"]

        return processed
    
    def __len__(self):
        return len(self.encoder_ids)

    def __getitem__(self, idx):
        encoder_ids = torch.tensor(self.encoder_ids[idx], dtype=torch.long)
        encoder_mask = torch.tensor(self.encoder_mask[idx], dtype=torch.long)
        initial_decoder_ids = torch.tensor(self.initial_decoder_ids[idx], dtype=torch.long)

        if self.split == "test":
            return encoder_ids, encoder_mask, initial_decoder_ids

        decoder_ids = torch.tensor(self.decoder_ids[idx], dtype=torch.long)
        return encoder_ids, encoder_mask, decoder_ids, initial_decoder_ids

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
    encoder_ids, encoder_masks, decoder_ids, initial_decoder_inputs = zip(*batch)

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_masks, batch_first=True, padding_value=0)
    decoder_ids = pad_sequence(decoder_ids, batch_first=True, padding_value=PAD_IDX)
    initial_decoder_inputs = torch.stack(initial_decoder_inputs, dim=0)

    decoder_inputs = torch.cat([initial_decoder_inputs, decoder_ids[:, :-1]], dim=1)
    decoder_targets = decoder_ids

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

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
    encoder_ids, encoder_masks, initial_decoder_inputs = zip(*batch)

    encoder_ids = pad_sequence(encoder_ids, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = pad_sequence(encoder_masks, batch_first=True, padding_value=0)
    initial_decoder_inputs = torch.stack(initial_decoder_inputs, dim=0)

    return encoder_ids, encoder_mask, initial_decoder_inputs

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
    train_x = load_lines(os.path.join(data_folder, "train.nl"))
    train_y = load_lines(os.path.join(data_folder, "train.sql"))
    dev_x = load_lines(os.path.join(data_folder, "dev.nl"))
    dev_y = load_lines(os.path.join(data_folder, "dev.sql"))
    test_x = load_lines(os.path.join(data_folder, "test.nl"))
    return train_x, train_y, dev_x, dev_y, test_x
