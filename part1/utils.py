import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation
    # text = example["text"]
    # words = text.split()
    # new_words = []
    # for word in words:
    #     # typo 
    #     if len(word) > 3 and random.random() < 0.4: # randomness for less acc
    #         chars = list(word)
    #         idx = random.randint(0, len(chars) - 2)
    #         chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    #         word = "".join(chars)
    #     new_words.append(word)
    # example["text"] = " ".join(new_words)
    

    # raise NotImplementedError

    text = example["text"]
    words = text.split()
    new_words = []

    vowel_map = {
        'a': 'e', 'e': 'i', 'i': 'o', 'o': 'u', 'u': 'a'
    }

    for word in words:
        w = word

        # swap 
        if len(w) > 3 and random.random() < 0.45:
            chars = list(w)
            idx = random.randint(0, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            w = "".join(chars)

        #vowel 
        if random.random() < 0.30:
            chars = list(w)
            for i, c in enumerate(chars):
                if c.lower() in vowel_map and random.random() < 0.5:
                    new_c = vowel_map[c.lower()]
                    chars[i] = new_c.upper() if c.isupper() else new_c
            w = "".join(chars)
        # drop filler
        if w.lower() in ["the", "a", "an", "is", "was", "to", "of"] and random.random() < 0.25:
            continue

        new_words.append(w)

    example["text"] = " ".join(new_words)

    ##### YOUR CODE ENDS HERE ######

    return example
