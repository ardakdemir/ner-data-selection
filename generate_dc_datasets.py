import os
import sys
import numpy as np
import json

ROOT_FOLDER = "/Users/ardaakdemir/bioMLT_folder/biobert_data/datasets/BioNER_2804_labeled_cleaned"
SAVE_FOLDER = "/Users/ardaakdemir/bioMLT_folder/biobert_data/datasets/BioNER_2505_DC_datasets"

bio_datasets = ['s800', 'NCBI-disease', 'JNLPBA', 'linnaeus', 'BC4CHEMD', 'BC2GM', 'BC5CDR']
news_datasets = ["conll-eng"]
file_names = {"train": "ent_train.tsv",
              "dev": "ent_devel.tsv"}


def get_dc_datasets(ROOT_FOLDER, split=0.80):
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

    for in_domain, out_domain, o_label in [[bio_datasets, news_datasets, "News"], [news_datasets, bio_datasets, "Bio"]]:
        for d in in_domain:
            in_domain = domain_datasets[d]["dev"]
            in_domain_size = len(in_domain)
            print("{} indomain sentences for {}".format(in_domain_size, d))
            out_domain_size = 5 * in_domain_size
            out_domain_dataset = []
            for od in out_domain:
                out_domain_dataset.extend(domain_datasets[od]["train"])
            np.random.shuffle(out_domain_dataset)
            out_domain_dataset = out_domain_dataset[:out_domain_size]
            print("OOD size: {}".format(len(out_domain_dataset)))
            dc_dataset = in_domain + out_domain_dataset
            labels = [d for _ in range(len(in_domain))] + [o_label for _ in range(len(out_domain_dataset))]
            dc_dataset = list(zip(dc_dataset, labels))
            np.random.shuffle(dc_dataset)
            split_index = int(len(dc_dataset) * split)
            sentences, labels = list(zip(*dc_dataset))
            dc_datasets[d] = {"train": {"sentences": sentences[:split_index], "labels": labels[:split_index]},
                              "test": {"sentences": sentences[split_index:], "labels": labels[split_index:]}}
            save_folder = os.path.join(SAVE_FOLDER, d)
            if not os.path.isdir(save_folder):
                os.makedirs(save_folder)
            for x, data in dc_datasets[d].items():
                save_path = os.path.join(save_folder, "{}.json".format(x))
                with open(save_path, "w") as j:
                    json.dump(data, j)


get_dc_datasets(ROOT_FOLDER, split=0.80)
