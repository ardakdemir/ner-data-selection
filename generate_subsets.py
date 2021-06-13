import os
import subprocess
import numpy as np
import argparse
from copy_devtest import copy_devtest
import pickle
import torch

SRC_FOLDER_ROOT = "../biobert_data/datasets/BioNER_2804_labeled_cleaned"


def generate_subsets(selected_file, save_folder_root, size, file_name="ent_train.tsv"):
    selected_data = pickle.load(open(selected_file, "rb"))
    for model_name, datasets in selected_data.items():
        for dataset, data in datasets.items():
            selected_sentences = data["selected_data"]
            my_data = selected_sentences[:size]
            save_folder = os.path.join(save_folder_root, model_name, dataset)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, file_name)
            s = "\n\n".join(
                ["\n".join(["{}\t{}".format(t, l) for t, l in zip(sent[-2], sent[-1])]) for sent in my_data])
            with open(save_path, "w") as o:
                o.write(s)
    copy_devtest(SRC_FOLDER_ROOT, save_folder_root, model_list=selected_data.keys())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str,
                        default='/home/aakdemir/biobert_data/datasets/BioNER_2804_labeled_cleaned',
                        help='')
    parser.add_argument('--save_folder', type=str,
                        default='/home/aakdemir/lda_selecteddata_1306',
                        help='')
    parser.add_argument('--selected_file_path', type=str,
                        default='/home/aakdemir/lda_save_1106/selected_pickle.p',
                        help='')
    parser.add_argument('--size', type=int, default=100)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root, save, size, selected_file_path = args.root_folder, args.save_folder, args.size, args.selected_file_path
    generate_subsets(selected_file_path, save, size, file_name="ent_train.tsv")


if __name__ == "__main__":
    main()
