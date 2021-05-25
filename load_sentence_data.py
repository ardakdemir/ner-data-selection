import os
import sys
import h5py
import pickle
import numpy as np
import torch
from collections import defaultdict


def load_sentence_data(all_sentences_path, model_name):
    all_sentences_dict = load_pickle(all_sentences_path)[model_name]
    final_dict = defaultdict(list)
    for sent in all_sentences_dict:
        label, vec, tokens, _ = sent
        final_dict[label].append((label, vec, " ".join(tokens)))
    for k, l in final_dict.items():
        print("{} sentences for {}".format(len(l), k))
        print("First sentence {}".format(l[0]))
    return final_dict


def load_pickle(path):
    return pickle.load(open(path, "rb"))


def main():
    file_path = ""
    to_do = 1


if __name__ == "__main__":
    main()
