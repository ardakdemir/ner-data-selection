import torch
import numpy as np
import time
from tqdm import tqdm
import argparse
import json
import h5py
import os
from torch.utils.data import Dataset, DataLoader
from transformers import *
from collections import defaultdict
from itertools import product
import logging
import utils
from vocab import Vocab
from gensim.models import FastText, KeyedVectors

CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"

special_tokens = [CLS_TOKEN, SEP_TOKEN]


def get_vocab(tokens):
    w2ind = {}
    for s in special_tokens:
        w2ind[s] = len(w2ind)
    for sent in tokens:
        for tok in sent:
            if tok not in w2ind:
                w2ind[tok] = len(w2ind)
    return w2ind


def get_bert_labels(tokens, labels):
    prev_label = "O"
    i = 0
    bert_labels = []
    for t in tokens:
        if t[:2] == "##":
            bert_labels.append(prev_label)
        elif t in special_tokens:
            bert_labels.append(t)
        else:
            label = labels[i]
            bert_labels.append(label)
            i += 1
            prev_label = label
    return bert_labels


class NerDataset(Dataset):
    def __init__(self, file_path,  tokenizer,size=None):
        sentences, labels = utils.read_ner_dataset(file_path, size=size)
        token_vocab, label_vocab = get_vocab(sentences), get_vocab(labels)
        self.token_vocab = Vocab(token_vocab)
        self.label_vocab = Vocab(label_vocab)

    def __len__(self):
        return len(self.sentences)

    def __getitem(self, index):
        sentence = " ".join(self.sentences[index])
        labels = self.labels[index]
        inputs = tokenizer(sentence, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = get_bert_labels(tokens, labels)
        labels = self.label_vocab.map(labels)
        labels = torch.tensor(labels).unsqueeze(0)
        return inputs, labels
