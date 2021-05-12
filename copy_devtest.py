import os
import subprocess
import sys

file_names = ["ent_devel.tsv", "ent_test.tsv"]
dataset_list = ['s800', 'NCBI-disease', 'JNLPBA', 'linnaeus', 'BC4CHEMD', 'BC2GM', 'BC5CDR', 'conll-eng']
model_list = ['robertaLarge', 'distilbertBaseUncased', 'BioBERT', 'BioWordVec']


def copy_devtest(src_folder_root, save_folder_root, model_list=model_list):
    for model in model_list:
        my_folder = os.path.join(save_folder_root, model)
        for d in dataset_list:
            src = os.path.join(src_folder_root, d)
            dest = os.path.join(my_folder, d)
            if not os.path.isdir(dest): os.makedirs(dest)
            for f in file_names:
                s = os.path.join(src, f)
                d = os.path.join(dest, f)
                cmd = "cp {} {}".format(s, d)
                subprocess.call(cmd, shell=True)


def main():
    src_folder_root = sys.args[1]
    save_folder_root = sys.args[2]
    copy_devtest(src_folder_root, save_folder_root)


if __name__ == "__main__":
    main()
