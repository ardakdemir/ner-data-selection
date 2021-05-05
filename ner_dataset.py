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
    def __init__(self, dataset, tokenizer, batch_size=5):
        self.dataset = dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        inps = []
        labs = []
        final_labels = []
        for b in range(self.batch_size):
            index = (index + b) % len(self.dataset)
            tokens, labels = self.dataset[index]
            inps.append(" ".join(tokens))
            labs.append(labels)

        inputs = self.tokenizer(inps, return_tensors="pt", padding=True)
        all_tokens = []
        for j, lab in enumerate(labs):
            input_tokens = inputs["input_ids"][j]
            tokens = self.tokenizer.convert_ids_to_tokens(input_tokens)
            all_tokens.append(tokens)
            l = get_bert_labels(tokens, lab)
            l = self.dataset.label_vocab.map(l)
            #             l = torch.tensor(l).unsqueeze(0)
            final_labels.append(l)
        return inputs, torch.tensor(final_labels), all_tokens
