import torch
import numpy as np
import time
from tqdm import tqdm
import argparse
import json
import h5py
import os
from torch.utils.data import Dataset, DataLoader
# from transformers import AutoTokenizer, AutoModel, BertForPreTraining, BertForTokenClassification, BertConfig
from transformers import *
from itertools import product
import logging
import utils

MODELS = [(RobertaModel, RobertaTokenizer, 'roberta-base', "robertaBase"),
          (RobertaModel, RobertaTokenizer, 'roberta-large', "robertaLarge"),
          (BertModel, BertTokenizer, 'bert-base-uncased', "bertBaseUncased"),
          (BertModel, BertTokenizer, 'bert-large-cased', "bertLargeCased"),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased', "distilbertBaseUncased"),
          (BertModel, BertTokenizer, "dmis-lab/biobert-v1.1", "BioBERT")]

ROOT_FOLDER = "/home/aakdemir/biobert_data/datasets/BioNER_2804"
SAVE_FOLDER = "/home/aakdemir/all_encoded_vectors_2804"


def encode_with_models(datasets, models_to_use, save_folder):
    """

    :param lines:
    :param models:
    :return:
    """
    dataset_to_model_to_states = {}
    for dataset_name, dataset in tqdm(datasets, desc="Datasets"):
        model_to_states = {}
        for model_class, tokenizer_class, model_name, save_name in tqdm(MODELS, desc="Models"):
            if model_name not in models_to_use: continue
            # Load pretrained model/tokenizer
            tokenizer = tokenizer_class.from_pretrained(model_name)
            model = model_class.from_pretrained(model_name)
            model.to(torch.device('cuda'))
            model_to_states[save_name] = {"sents": [], "states": []}
            # Encode text
            start = time.time()
            for sentence in tqdm(dataset,desc="sentences.."):
                model_to_states[save_name]['sents'].append(sentence)
                input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True,
                                                           truncate=True,
                                                           max_length=128)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
                input_ids = input_ids.to(torch.device('cuda'))
                with torch.no_grad():
                    output = model(input_ids)
                    last_hidden_states = output[0]

                    # avg pool last hidden layer
                    squeezed = last_hidden_states.squeeze(dim=0)
                    masked = squeezed[:input_ids.shape[1], :]
                    avg_pooled = masked.mean(dim=0)
                    model_to_states[save_name]['states'].append(avg_pooled.cpu())
            end = time.time()
            t = round(end - start)
            print('Encoded {}  with {} in {} seconds'.format(dataset_name, model_name, t))
            np_tensors = [np.array(tensor) for tensor in model_to_states[save_name]['states']]
            # model_to_states[model_name]['states'] = np.stack(np_tensors)
            if not os.path.isdir(save_folder):os.makedirs(save_folder)
            save_path = os.path.join(save_folder, "{}_{}_vectors.h5".format(dataset_name, save_name))
            with h5py.File(save_path, "w") as h:
                h["vectors"] = np.stack(np_tensors)
                h["time"] = [t]
        dataset_to_model_to_states[dataset_name] = model_to_states
    return dataset_to_model_to_states


def main():
    folder = ROOT_FOLDER
    save_folder = SAVE_FOLDER
    size = 3000
    models_to_use = [x[2] for x in MODELS[-1:]]
    datasets = utils.get_sentence_datasets_from_folder(folder, size=size, file_name="ent_train.tsv")
    for n, d in datasets:
        print("{} size {}".format(n, len(d)))
    dataset_to_model_to_states = encode_with_models(datasets, models_to_use, save_folder)


if __name__ == "__main__":
    main()
