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
        for dataset, data in datasets.values():
            selected_data = data["selected_data"]
            my_data = selected_data[:size]
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
                        default='/Users/ardaakdemir/bioMLT_folder/biobert_data/datasets/EntNER_2504',
                        help='training file for sa')
    parser.add_argument('--save_folder', type=str,
                        default='/Users/ardaakdemir/bioMLT_folder/biobert_data/datasets/small_EntNER_2504',
                        help='validation file for sa')
    parser.add_argument('--selected_file_path', type=str,
                        default='/home/aakdemir/dataselection_0606_labeled/selected_pickle.p',
                        help='validation file for sa')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--size', type=int, default=500)
    return args


def main():
    args = parser.parse_args()
    root, save, size, selected_file_path = args.root_folder, args.save_folder, args.size, args.selected_file_path
    generate_subsets(selected_file_path, save, size, file_name="ent_train.tsv")

if __name__ == "__main__":
    main()
