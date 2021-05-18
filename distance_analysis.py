import sys
import os
import matplotlib.pyplot as plt
import time
import pickle
from numpy import dot, inner
from numpy.linalg import norm
import numpy as np


def cos_similarity(a, b):
    return inner(a, b) / (norm(a) * norm(b))


def get_similarity(source_examples, ref_vecs):
    sims = []
    sample_size = 500
    for s in source_examples:
        np.random.shuffle(ref_vecs)
        sample_vecs = ref_vecs[:sample_size]
        my_sim = max([cos_similarity(s[1], v) for v in sample_vecs])
        sims.append(my_sim)
    return sims


def distance_analysis(selected, all_sents, save_folder):
    domain_to_sents = selected["BioBERT"]
    all_vectors = [instance[1] for instance in all_sents["BioBERT"]]
    for k, d in domain_to_sents.items():
        beg = time.time()
        s = data["selected_data"][0]
        ref_vecs = data["all_target_data"]["states"]
        sims = get_similarity(data["selected_data"], ref_vecs)
        sims.sort(reverse=True)
        plt.figure(figsize=(12, 8))
        plt.xlabel("Cosine simillarity")
        plt.ylabel("Number of instances")
        plt.hist(sims, bins=100)
        save_path = os.path.join(save_folder, "cosinesim_hist_{}.png".format(k))
        plt.savefig(save_path)
        end = time.time()
        tt = round(end - beg, 3)
        print("{} seconds for {}... ".format(tt, k))


def main():
    args = sys.argv
    selected_file = args[1]
    all_sentences_file = args[2]
    save_folder = args[3]
    all_sentences = pickle.load(open(all_sentences_file, "rb"))
    selected_sentences = pickle.load(open(selected_file, "rb"))
    distance_analysis(selected_sentences, all_sentences, save_folder)


if __name__ == "__main__":
    main()
