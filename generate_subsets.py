import os
import subprocess
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, default='/Users/ardaakdemir/bioMLT_folder/biobert_data/datasets/EntNER_2504',
                        help='training file for sa')
    parser.add_argument('--save_folder', type=str, default='/Users/ardaakdemir/bioMLT_folder/biobert_data/datasets/small_EntNER_2504',
                        help='validation file for sa')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--size', type=int, default=500)
    return args

def generate_small_datasets( file_name)
def main():
    args = parser.parse_args()
    root,save = args.root_folder,args.save


if __name__ == "__main__":
    main()
