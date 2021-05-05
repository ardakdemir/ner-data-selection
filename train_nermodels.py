import torch
import numpy as np
import time
from tqdm import tqdm
import argparse
import json
import h5py
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from transformers import *
from collections import defaultdict
from itertools import product
import logging
from gensim.models import FastText, KeyedVectors
from conll_eval import evaluate_conll_file

from gensim.utils import tokenize

MODELS = [(RobertaModel, RobertaTokenizer, 'roberta-large', "robertaLarge"),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased', "distilbertBaseUncased"),
          (BertModel, BertTokenizer, "dmis-lab/biobert-v1.1", "BioBERT")]

ROOT_FOLDER = "/home/aakdemir/biobert_data/datasets/BioNER_2804"
SAVE_FOLDER = "/home/aakdemir/all_encoded_vectors_0305"
CONLL_SAVE_PATH = "conll_output_0505.txt"
BioWordVec_FOLDER = "../biobert_data/bio_embedding_extrinsic"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# train_file_path = "/Users/ardaakdemir/bioMLT_folder/biobert_data/datasets/BioNER_2804/BC2GM/ent_train.tsv"
# dev_file_path = "/Users/ardaakdemir/bioMLT_folder/biobert_data/datasets/BioNER_2804/BC2GM/ent_devel.tsv"

def parse_args():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Working  on {}".format(device))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_file_path", default="../biobert_data/datasets/BioNER_2804/BC2GM/ent_train.tsv", type=str,
        required=False
    )
    parser.add_argument(
        "--dev_file_path", default="../biobert_data/datasets/BioNER_2804/BC2GM/ent_devel.tsv", type=str, required=False
    )
    parser.add_argument(
        "--test_file_path", default="../biobert_data/datasets/BioNER_2804/BC2GM/ent_test.tsv", type=str, required=False
    )
    parser.add_argument(
        "--save_folder", default="../dataselect_nerresult_0505", type=str, required=False,
        help="The path to save everything..."
    )
    parser.add_argument(
        "--input_dims", default=768, type=int, required=False,
    )
    parser.add_argument(
        "--size", default=100, type=int, required=False,
    )
    parser.add_argument(
        "--output_dim", default=6, type=int, required=False,
    )
    parser.add_argument(
        "--batch_size", default=12, type=int, required=False,
    )
    args = parser.parse_args()
    args.device = device
    return args


def train(args):
    #     biobert_model_tuple = MODELS[-1]
    model_tuple = (BertModel, BertTokenizer, "bert-base-uncased", "Bert-base")
    dataset_loaders = {}

    save_folder = args.save_folder
    size = args.size
    batch_size = args.batch_size
    train_file_path = args.train_file_path
    dev_file_path = args.dev_file_path

    model_name = model_tuple[2]
    tokenizer = BertTokenizer.from_pretrained(model_name)

    ner_dataset = NerDataset(train_file_path, size=size)
    dataset_loader = NerDatasetLoader(ner_dataset, tokenizer, batch_size=batch_size)
    dataset_loaders["train"] = dataset_loader
    num_classes = len(dataset_loader.dataset.label_vocab)
    args.output_dim = num_classes

    eval_ner_dataset = NerDataset(dev_file_path, size=size)
    eval_ner_dataset.label_vocab = ner_dataset.label_vocab
    eval_ner_dataset.token_vocab = ner_dataset.token_vocab
    eval_dataset_loader = NerDatasetLoader(eval_ner_dataset, tokenizer, batch_size=batch_size)
    dataset_loaders["devel"] = eval_dataset_loader

    model = NerModel(args, model_tuple)
    train_model(model, dataset_loaders, save_folder)


def write_to_conll_format(conll_data, save_path):
    s = ""
    for sent in conll_data:
        for t, l, p in sent:
            if t[:2] == "##": continue
            if t in ["[CLS]", "[SEP]"]: continue
            if t == "[PAD]": break

            s += "{}\t{}\t{}\n".format(t, l, p)
        s += "\n"
    with open(save_path, "w") as o:
        o.write(s)


def evaluate(model, dataset_loader, save_path):
    model = model.eval()
    preds = labels = []
    conll_data = []
    for i in tqdm(range(len(dataset_loader)), desc="evaluation"):
        with torch.no_grad():
            inputs, label, tokens = dataset_loader[i]
            output = model(inputs)
            b, n, c = output.shape
            for l in label:
                labels.extend(l.detach().cpu().tolist())
            pred = torch.argmax(output, dim=2)
            for p in pred:
                preds.extend(p.detach().cpu().tolist())
            for t, p, l in zip(tokens, pred.detach().cpu().tolist(), label.detach().cpu().tolist()):
                conll_data.append(list(zip(t, l, p)))
        if i > 5: break
    print("Conll data", conll_data[:10])
    write_to_conll_format(conll_data, save_path)
    pre, rec, f1 = evaluate_conll_file(open(save_path).readlines())
    print("Pre: {} Rec: {} F1: {}".format(pre, rec, f1))

    print("{} preds {} labels...".format(len(preds), len(labels)))
    print("Preds: {}  labels: {}".format(preds[:5], labels[:5]))
    return pre, rec, f1


def train_model(model, dataset_loaders, save_folder):
    epoch_num = 2
    # eval_interval = len(dataset_loader)
    eval_interval = 5
    model.to(device)
    model = model.train()
    optimizer = AdamW(model.parameters())
    pad_index = dataset_loaders.dataset.label_vocab.w2ind["[PAD]"]
    criterion = CrossEntropyLoss(ignore_index=pad_index)

    train_loader = dataset_loaders["train"]
    eval_loader = dataset_loaders["devel"]
    for j in tqdm(range(epoch_num), desc="Epochs"):
        model = model.train()
        for i in tqdm(range(eval_interval), desc="training"):
            optimizer.zero_grad()
            inputs, label, tokens = train_loader[i]
            output = model(inputs)
            b, n, c = output.shape
            output = output.reshape(b, c, n)
            print("Model output shape {}".format(output.shape))
            label = label.to(device)
            print("Label shape", label.shape)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            print(loss.item())
        res = evaluate(model, eval_loader, save_path)
        print("Result for epoch {} {} ".format(j, res))


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
