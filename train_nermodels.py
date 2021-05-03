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
from gensim.models import FastText, KeyedVectors

from gensim.utils import tokenize

MODELS = [(RobertaModel, RobertaTokenizer, 'roberta-large', "robertaLarge"),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased', "distilbertBaseUncased"),
          (BertModel, BertTokenizer, "dmis-lab/biobert-v1.1", "BioBERT")]

ROOT_FOLDER = "/home/aakdemir/biobert_data/datasets/BioNER_2804"
SAVE_FOLDER = "/home/aakdemir/all_encoded_vectors_0305"

BioWordVec_FOLDER = "../biobert_data/bio_embedding_extrinsic"


def parse_args():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Working  on {}".format(device))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dims", default=768, type=int,
        help="Output dimension of ner latent module"
    )

    parser.add_argument(
        "--output_dim", default=4, type=int,
        help="Output dimension of ner module"
    )

    args = parser.parse_args()
    return args


def train():
    biobert_model_tuple = MODELS[-1]
    args = parse_args()
    file_path = "biobert_data/datasets/BioNER_2804/s800/ent_test.tsv"
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

    ner_dataset = NerDataset(file_path,tokenizer,size = 100)
    print(ner_dataset[0])
    NerModel(args, biobert_model_tuple)


def main():

    train()


if __name__ == "__main__":
    main()
