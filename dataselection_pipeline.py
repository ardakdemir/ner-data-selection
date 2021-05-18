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
        "--dataset_root", default="/home/aakdemir/dataselection_1805_labeled", type=str,
        required=False
    )
    parser.add_argument(
        "--evaluate_root", default="/home/aakdemir/dataselection_1805_labeled", type=str,
        required=False
    )
    parser.add_argument(
        "--save_folder", default="/home/aakdemir/all_encoded_vectors_1805", type=str, required=False)
    parser.add_argument(
        "--dev_save_folder", default="/home/aakdemir/all_dev_encoded_vectors_1805", type=str, required=False)
    parser.add_argument(
        "--selected_save_root", default="/home/aakdemir/dataselection_1805_labeled", type=str, required=False)
    parser.add_argument(
        "--random", default=False, action="store_true", required=False)
    parser.add_argument(
        "--save_folder_root", default="../dataselect_nerresult_1805", type=str, required=False,
        help="The path to save everything..."
    )
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
    parser.add_argument(
        "--class_dict_path", default="../dataselect_nerresult_0505/class_to_idx.json", type=str, required=False,
        help="The path to save everything..."
    )
    parser.add_argument(
        "--target_dataset_path", default="../biobert_data/datasets/BioNER_2804/BC2GM", type=str, required=False)
    parser.add_argument(
        "--input_dims", default=768, type=int, required=False,
    )
    parser.add_argument(
        "--size", default=-1, type=int, required=False,
    )
    parser.add_argument(
        "--eval_interval", default=-1, type=int, required=False,
    )
    parser.add_argument(
        "--epoch_num", default=10, type=int, required=False,
    )
    parser.add_argument(
        "--output_dim", default=6, type=int, required=False,
    )
    parser.add_argument(
        "--batch_size", default=12, type=int, required=False,
    )
    parser.add_argument(
        "--multiple", default=False, action="store_true", help="Run for all datasets"
    )
    parser.add_argument(
        "--multi_model", default=False, action="store_true", help="Run for all model"
    )
    parser.add_argument(
        "--inference", default=False, action="store_true", help="Run inference only."
    )
    parser.add_argument(
        "--dev_only", default=False, action="store_true", help="If True, only uses the dev split for training"
    )
    args = parser.parse_args()
    args.device = device
    return args


def select_data(models_to_use, dataset_list, args):
    select_store_data(models_to_use, dataset_list, args)


def train_model(save_folder_root, dataset_list, args):
    train_all_datasets(save_folder_root, dataset_list, args)


def hyperparameter_search():
    models_to_use = [model_tuple[2]]
    dataset_list = ["BC2GM"]
    select_sizes = [5000, 10000, 20000, 30000]



def main():
    args = parse_args()
    models_to_use = [model_tuple[2]]
    dataset_list = ["BC2GM", "s800"]
    save_folder_root = args.save_folder_root
    args.dataset_root = args.selected_save_root
    args.evaluate_root = args.selected_save_root
    print("Dataset root {} eval root {} Save to : {}".format(args.dataset_root, args.evaluate_root, save_folder_root))
    # select_store_data(models_to_use, dataset_list, args)
    train_all_datasets(save_folder_root, dataset_list, args)


if __name__ == "__main__":
    main()
