import os
import sys
import numpy as np
import json

ROOT_FOLDER = "/home/aakdemir/biobert_data/datasets/BioNER_2804_labeled_cleaned"

bio_datasets = ['s800', 'NCBI-disease', 'JNLPBA', 'linnaeus', 'BC4CHEMD', 'BC2GM', 'BC5CDR']
news_datasets = ["conll-eng"]
all_datasets = bio_datasets + news_datasets
file_names = {"train": "ent_train.tsv",
              "dev": "ent_devel.tsv"}


def get_out_domain_data(domain_datasets, indomain_name, size):
    out_domain_dataset = []
    all_od_datasets = []
    for x in domain_datasets:
        if x == indomain_name: continue
        all_od_datasets.extend(domain_datasets[x]["train"])
    print("Out of domain {} datasets in total... ".format(len(all_od_datasets)))
    np.random.shuffle(all_od_datasets)
    all_od_datasets = all_od_datasets[:size]
    return all_od_datasets


def get_dc_datasets(ROOT_FOLDER, split=0.80, oov_rel_size=5):
    domain_datasets = {}
    dc_datasets = {}
    for d in bio_datasets + news_datasets:
        domain_datasets[d] = {}
        for k, f in file_names.items():
            p = os.path.join(ROOT_FOLDER, d, f)
            with open(p, "r") as o:
                dataset = o.read().split("\n\n")
                sentences = [" ".join([t.split()[0] for t in sent.split("\n") if len(sent.split("\n")) > 0]) for sent in
                             dataset]
                domain_datasets[d][k] = sentences
    print("Domain dataset keys: {}".format(domain_datasets.keys()))
    for in_domain_name in all_datasets:
        o_label = "OTHER_DOMAIN"
        in_domain_dataset = domain_datasets[d]["dev"]
        in_domain_size = len(in_domain_dataset)
        print("{} indomain sentences for {}".format(in_domain_size, in_domain_name))
        out_domain_size = oov_rel_size * in_domain_size
        out_domain_dataset = get_out_domain_data(domain_datasets, in_domain_name, out_domain_size)
        np.random.shuffle(out_domain_dataset)
        out_domain_dataset = out_domain_dataset[:out_domain_size]
        print("OOD size: {}".format(len(out_domain_dataset)))
        dc_dataset = in_domain_dataset + out_domain_dataset
        labels = [in_domain_name for _ in range(len(in_domain_dataset))] + [o_label for _ in
                                                                            range(len(out_domain_dataset))]
        dc_dataset = list(zip(dc_dataset, labels))
        np.random.shuffle(dc_dataset)
        split_index = int(len(dc_dataset) * split)
        sentences, labels = list(zip(*dc_dataset))
        dc_datasets[in_domain_name] = {
            "train": {"sentences": sentences[:split_index], "labels": labels[:split_index]},
            "test": {"sentences": sentences[split_index:], "labels": labels[split_index:]}}
        save_folder = os.path.join(SAVE_FOLDER, in_domain_name)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        for x, data in dc_datasets[in_domain_name].items():
            save_path = os.path.join(save_folder, "{}.json".format(x))
            with open(save_path, "w") as j:
                json.dump(data, j)


for rel_size in [1,2,3,5,10]:
    SAVE_FOLDER = "/home/aakdemir/biobert_data/datasets/BioNER_2505_DC_datasets_relsize_{}".format(
        rel_size)
    get_dc_datasets(ROOT_FOLDER, split=0.80, oov_rel_size=rel_size)
