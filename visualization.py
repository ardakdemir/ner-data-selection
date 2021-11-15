import h5py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
# Plot confusion matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle

import matplotlib.pyplot as plt
# import plotly.graph_objects as go

file_names = {"train": "ent_train.tsv",
              "dev": "ent_devel.tsv",
              "test": "ent_test.tsv"}
colors = ["b", "g", "r", "c", "m", "y", "k", "tab:orange"]
root_folder = "biobert_data/datasets/BioNER_2804"
markers = [".", "v", "+", "x", "<", ">", "D", "2"]

vector_index = 1
dataset_list = ['s800', 'NCBI-disease', 'JNLPBA', 'linnaeus', 'BC4CHEMD', 'BC2GM', 'BC5CDR', 'conll-eng']

all_sentences_pickle = "/Users/ardaakdemir/dataselection_data/instanceselection_2005_labeled/allsentences_pickle.p"
selected_pickle = "/Users/ardaakdemir/dataselection_data/instanceselection_2005_labeled/selected_pickle.p"




all_sentences = pickle.load(open(all_sentences_pickle, "rb"))
selected_sentences = pickle.load(open(selected_pickle, "rb"))
selected_save_folder = "../biobert_selected_1506"
m = "BioBERT"


def get_sent_nums(root, file_names=file_names):
    return {k: len(open(os.path.join(root, v)).read().split("\n\n")) for k, v in file_names.items()}


def get_dataset_sizes(root_folder):
    return {c: get_sent_nums(os.path.join(root_folder, c)) for c in os.listdir(root_folder) if
            os.path.isdir(os.path.join(root_folder, c))}


def load_vectors(p):
    with h5py.File(p, "r") as h:
        return h["vectors"][:]


def load_vectors_from_folder(folder):
    vect_dict = {}
    for x in os.listdir(folder):
        name = x.split(".")[0]
        p = os.path.join(folder, x)
        vectors = load_vectors(p)
        vect_dict[name] = vectors
    return vect_dict


def visualize_selected():
    if not os.path.isdir(selected_save_folder): os.makedirs(selected_save_folder)
    select_size = 5000
    for i, d in enumerate(dataset_list):
        all_target_vectors = [x.numpy() for x in selected_sentences[m][d]["all_target_data"]["states"]]
        all_model_vectors = [s[vector_index].numpy() for s in all_sentences[m]]
        all_selected_vectors = [x[1].numpy() for x in selected_sentences[m][d]["selected_data"]]
        sample_size = 30000
        np.random.shuffle(all_model_vectors)
        print("{} == All source size: {} target size : {}  selected size : {}".format(d, len(
            all_model_vectors[:sample_size]),
                                                                                      len(all_target_vectors), len(
                all_selected_vectors[:select_size])))
        v = np.array(all_model_vectors[:sample_size] + all_target_vectors + all_selected_vectors[:select_size])
        print("V shape: {}".format(v.shape))
        pca = PCA(n_components=2)
        pca.fit(v)
        selected_pca = pca.transform(all_selected_vectors[:select_size])
        target_pca = pca.transform(all_target_vectors)
        source_pca = pca.transform(all_model_vectors[:sample_size])
        plt.figure(figsize=(12, 8))
        plt.title(d if d != "conll-eng" else "News", fontsize=25)
        plt.scatter(source_pca[:, 0],
                    source_pca[:, 1],
                    color="grey",
                    marker=markers[0],
                    label="general domain", alpha=0.3)
        plt.scatter(selected_pca[:, 0],
                    selected_pca[:, 1],
                    color="tab:orange",
                    marker=markers[2],
                    label="selected", alpha=0.9)
        plt.scatter(target_pca[:, 0],
                    target_pca[:, 1],
                    color="green",
                    marker=markers[1],
                    label="in-domain",
                    alpha=0.4)
        plt.tight_layout()
        if i == 0:
            plt.legend(markerscale=3, fontsize=25)
        s = os.path.join(selected_save_folder, "selected_{}.png".format(d))
        plt.savefig(s)
        # plt.show()


def plot_sizes():
    if not os.path.isdir(selected_save_folder): os.makedirs(selected_save_folder)
    all_root_folder = "/Users/ardaakdemir/bioMLT_folder/biobert_data/datasets/BioNER_2804_labeled_cleaned"
    dataset_sizes = get_dataset_sizes(all_root_folder)
    file_names = ["train", "dev", "test"]
    corpora_names = list(dataset_sizes.keys())
    fig = go.Figure(data=[
        go.Bar(name=t, x=[a if a != "conll-eng" else "News" for a in corpora_names],
               width=[0.2 for _ in range(len(corpora_names))], y=[dataset_sizes[c][t] for c in corpora_names]) for t
        in file_names])
    fig.update_layout(barmode='group', yaxis_title="Number of sentences")
    fig.write_image(os.path.join(selected_save_folder, "dataset_sizes.png"))
    # fig.show()


def main():
    visualize_selected()
    # plot_sizes()


if __name__ == "__main__":
    main()
