import numpy as np
import argparse
from transformers import BertTokenizer, BertForTokenClassification
import torch
import json
import h5py
import os
from torch.utils.data import Dataset, DataLoader
from transformers import *
from collections import defaultdict
import logging

CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"

special_tokens = [CLS_TOKEN, SEP_TOKEN, UNK_TOKEN, PAD_TOKEN]


def read_ner_dataset(file_path, size=None):
    dataset = open(file_path).read().split("\n\n")
    sentences = [[x.split()[0] for x in sent.split("\n") if len(x.split()) > 0] for sent in dataset]
    labels = [[x.split()[-1] for x in sent.split("\n") if len(x.split()) > 0] for sent in dataset]
    data = list(zip(sentences, labels))
    if not size:
        return list(zip(*data))
    else:
        np.random.shuffle(data)
        return list(zip(*data[:size]))
    import torch


class Vocab:

    def __init__(self, w2ind):
        self.w2ind = w2ind
        self.ind2w = [x for x in w2ind.keys()]

    def __len__(self):
        return len(self.w2ind)

    def map(self, units):
        return [self.w2ind.get(x, UNK_TOKEN) for x in units]

    def unmap(self, idx):
        return [self.ind2w[i] for i in idx]
    def set_w2ind(self,w2ind):
        self.w2ind = w2ind
        self.ind2w = [v:k for k,v in self.w2ind.items()]


def get_vocab(tokens):
    w2ind = {}
    for s in special_tokens:
        w2ind[s] = len(w2ind)
    for sent in tokens:
        for tok in sent:
            if tok not in w2ind:
                w2ind[tok] = len(w2ind)
    return w2ind


def get_bert_labels(tokens, labels, raw_tokens):
    prev_label = "O"
    i, k = 0, 0
    bert_labels = []
    curr_tok = ""
    while True:
        if i == len(tokens):
            break
        curr_tok += tokens[i] if tokens[i][:2] != "##" else tokens[i][2:]
        if curr_tok in special_tokens:
            bert_labels.append(curr_tok)
            curr_tok = ""
        elif curr_tok == raw_tokens[k]:
            bert_labels.append(labels[k])
            k += 1
            curr_tok = ""
        else:
            bert_labels.append(labels[k])
        i += 1
    return bert_labels


class NerDataset(Dataset):
    def __init__(self, file_path, size=None):
        sentences, labels = read_ner_dataset(file_path, size=size)
        token_vocab, label_vocab = get_vocab(sentences), get_vocab(labels)
        self.token_vocab = Vocab(token_vocab)
        self.label_vocab = Vocab(label_vocab)
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        tokens = self.sentences[index]
        labels = self.labels[index]
        return tokens, labels


class NerDatasetLoader:
    def __init__(self, dataset, tokenizer, batch_size=5, for_eval=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.for_eval = for_eval

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):
        inps = []
        labs = []
        raw_tokens = []
        final_labels = []
        if self.for_eval:
            index = index * self.batch_size
        else:
            index = np.random.randint(0, len(self.dataset) / self.batch_size)
        for b in range(self.batch_size):
            index = (index + b) % len(self.dataset)
            tokens, labels = self.dataset[index]
            inps.append(" ".join(tokens))
            labs.append(labels)
            raw_tokens.append(tokens)

        inputs = self.tokenizer(inps, return_tensors="pt", padding=True, truncation=True,
                                max_length=512)
        all_tokens = []
        for j, lab in enumerate(labs):
            input_tokens = inputs["input_ids"][j]
            tokens = self.tokenizer.convert_ids_to_tokens(input_tokens)
            all_tokens.append(tokens)
            l = get_bert_labels(tokens, lab, raw_tokens[j])
            l = self.dataset.label_vocab.map(l)
            #             l = torch.tensor(l).unsqueeze(0)
            final_labels.append(l)
        return inputs, torch.tensor(final_labels), all_tokens
