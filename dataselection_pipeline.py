import torch
import numpy as np
import time
from tqdm import tqdm
import argparse
import json
import h5py
import os
from numpy import dot, inner
from numpy.linalg import norm
from sklearn.decomposition import PCA
import matplotlib.pyplot as pltN
from torch.utils.data import Dataset, DataLoader
# from transformers import *
from transformers import RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer, BertModel, BertTokenizer
from write_selected_sentences import write_selected_sentences
from collections import defaultdict
from itertools import product
from copy_devtest import copy_devtest
from annotate_all_entities import annotate_all_entities
import logging
import utils
import pickle
from gensim.models import FastText, KeyedVectors
from generate_vectors import select_store_data
from train_nermodels import train_all_datasets
model_tuple = (BertModel, BertTokenizer, "dmis-lab/biobert-v1.1", "BioBERT")


def parse_args():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Working  on {}".format(device))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_folder", default="/home/aakdemir/biobert_data/datasets/BioNER_2804_labeled", type=str, required=False)
    parser.add_argument(
        "--dataset_name", default="random", type=str, required=False)
    parser.add_argument(
        "--save_folder", default="/home/aakdemir/all_encoded_vectors_1805", type=str, required=False)
    parser.add_argument(
        "--dev_save_folder", default="/home/aakdemir/all_dev_encoded_vectors_1805", type=str, required=False)
    parser.add_argument(
        "--selected_save_root", default="/home/aakdemir/dataselection_1805_labeled", type=str, required=False)
    parser.add_argument(
        "--random", default=False, action="store_true", required=False)
    parser.add_argument(
        "--repeat", default=4, type=int, required=False)
    parser.add_argument(
        "--selection_method", default="cosine_instance", choices=["cosine_instance", "cosine_subset"], required=False)
    parser.add_argument(
        "--select_size", default=200, type=int, required=False)
    parser.add_argument(
        "--train_size", default=100, type=int, required=False)
    parser.add_argument(
        "--dev_size", default=50, type=int, required=False)
    parser.add_argument(
        "--biowordvec_folder", default="/home/aakdemir/biobert_data/bio_embedding_extrinsic", type=str, required=False)
    parser.add_argument(
        "--word2vec_folder", default="/home/aakdemir/biobert_data/word2Vec/en/en.bin", type=str, required=False)
    parser.add_argument(
        "--cos_sim_sample_size", default=1000, type=int, required=False)
    args = parser.parse_args()
    args.device = device
    return args

def select_data(args):
    models_to_use = [model_tuple[2]]
    dataset_list = ["BC2GM","s800"]
    select_store_data(models_to_use, dataset_list,args)
    train_all_datasets(save_folder_root, dataset_list, args)
def train_model(args):
    y = 2

def main():
    args = parse_args()
    select_data(args)
    train_model(args)

if __name__ == "__main__":
    main()