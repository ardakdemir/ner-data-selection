import sys
import os
import matplotlib.pyplot as plt
import time
import pickle
from numpy import dot, inner
from numpy.linalg import norm
import numpy as np
from tqdm import tqdm
import torch.nn.CosineSimilarity as torch_cos_sim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cos_similarity(a, b):
    return inner(a, b) / (norm(a) * norm(b))


def get_similarity(source_examples, ref_vecs):
    sims = []
    sample_size = 1000
    for s in tqdm(source_examples,desc="Source examples"):
        np.random.shuffle(ref_vecs)
        sample_vecs = ref_vecs[:sample_size]
        my_sim = torch_cos_similarity(vec,sample_vecs)
        sims.append(my_sim)
    return sims

def torch_cos_similarity(vec,sample_vecs):
    torch_cosine_similarity  = torch_cos_sim(dim=1)
    sample_vecs = torch.tensor(sample_vecs)
    sample_vecs = sample_vecs.to(DEVICE)
    vec = torch.tensor(vec)
    vec = vec.to(DEVICE)
    vec.expand(len(sample_vecs),-1)
    print("Vec shape: {} sample shape {}".format(sample_vecs.shape,vec.shape))
    cos_sims = torch_cosine_similarity(vec,sample_vecs)
    return torch.max(cos_sims).item()

def distance_analysis(selected, all_sents, save_folder):
    domain_to_sents = selected["BioBERT"]
    all_vectors = all_sents["BioBERT"]
    for k, data in tqdm(domain_to_sents.items()):
        beg = time.time()
        s = data["selected_data"][0]
        ref_vecs = data["all_target_data"]["states"]
        sims = get_similarity(all_vectors, ref_vecs)
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
    all_sentences_file = args[1]
    selected_file = args[2]
    save_folder = args[3]
    if not os.path.isdir(save_folder):os.makedirs(save_folder)
    all_sentences = pickle.load(open(all_sentences_file, "rb"))
    selected_sentences = pickle.load(open(selected_file, "rb"))
    distance_analysis(selected_sentences, all_sentences, save_folder)


if __name__ == "__main__":
    main()
