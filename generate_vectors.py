import torch
import numpy as np
import time
from tqdm import tqdm
import argparse
import json
import h5py
import os
from torch.utils.data import Dataset, DataLoader
from transformers import *
from collections import defaultdict
from itertools import product
import logging
import utils
from gensim.models import FastText, KeyedVectors

MODELS = [(RobertaModel, RobertaTokenizer, 'roberta-large', "robertaLarge"),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased', "distilbertBaseUncased"),
          (BertModel, BertTokenizer, "dmis-lab/biobert-v1.1", "BioBERT")]

ROOT_FOLDER = "/home/aakdemir/biobert_data/datasets/BioNER_2804"
SAVE_FOLDER = "/home/aakdemir/small_encoded_vectors_0205"

BioWordVec_FOLDER = "../biobert_data/bio_embedding_extrinsic"


def get_w2v_sent_reps(dataset, model, max_pool=False):
    """
    Encodes the lines in a text file using word2vec
    """
    vecs = []
    toks = []
    for sent in sentences:
        vec, sent_toks = encode_sent_with_w2v(sent, model, max_pool)
        vecs.append(vec)
        toks.append(sent_toks)
    return np.stack(vecs), toks


def encode_sent_with_w2v(sent, model, max_pool=False):
    """
    Encodes a sentence as a sum of its corresponding word2vec embeddings.
    """
    MODEL_SIZE = 300
    toks = list(tokenize(sent))
    vecs = []
    for tok in toks:
        if tok in model:
            vecs.append(model[tok])
    if len(vecs):
        if max_pool:
            pooled = np.max(np.stack(vecs), axis=0)
        else:
            pooled = np.mean(np.stack(vecs), axis=0)
    else:
        pooled = model['unk']
    return pooled, toks


def encode_with_bioword2vec(datasets, save_folder):
    dataset_to_states = {}

    model = KeyedVectors.load_word2vec_format(BioWordVec_FOLDER, binary=True)
    for dataset_name, dataset in tqdm(datasets, desc="Datasets"):
        begin = time.time()
        vecs, toks = get_w2v_sent_reps(dataset, model, max_pool=False)
        dataset_to_states[dataset_name] = vecs
        end = time.time()
        t = round(end - begin, 3)
        save_fold = os.path.join(save_folder, "BioWordVec")
        if not os.path.isdir(save_fold):
            os.makedirs(save_fold)

        save_path = os.path.join(save_fold, "{}.h5".format(dataset_name))
        with h5py.File(save_path, "w") as h:
            h["vectors"] = vecs
            h["time"] = [t]
    return {"BioWordVec": dataset_to_states}


def encode_with_models(datasets, models_to_use, save_folder):
    """

    :param lines:
    :param models:
    :return:
    """
    model_to_domain_to_encodings = defaultdict(dict)
    for dataset_name, dataset in tqdm(datasets, desc="Datasets"):
        model_to_states = {}
        for model_class, tokenizer_class, model_name, save_name in tqdm(MODELS, desc="Models"):
            if model_name not in models_to_use:
                print("Skipping {}".format(model_name))
                continue
            # Load pretrained model/tokenizer
            tokenizer = tokenizer_class.from_pretrained(model_name)
            model = model_class.from_pretrained(model_name)
            model.to(torch.device('cuda'))
            model_to_states[save_name] = {"sents": [], "states": []}
            # Encode text
            start = time.time()
            i = 0
            for sentence in tqdm(dataset, desc="sentences.."):
                model_to_states[save_name]['sents'].append(sentence)
                if i == 0:
                    print(sentence)
                    i += 1
                input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True,
                                                           truncation=True,
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
            save_fold = os.path.join(save_folder, save_name)
            if not os.path.isdir(save_fold):
                os.makedirs(save_fold)
            save_path = os.path.join(save_fold, "{}.h5".format(dataset_name))
            with h5py.File(save_path, "w") as h:
                h["vectors"] = np.stack(np_tensors)
                h["time"] = [t]
        for k, d in model_to_states.items():
            model_to_domain_to_encodings[k][dataset_name] = d
    return model_to_domain_to_encodings


def main():
    folder = ROOT_FOLDER
    save_folder = SAVE_FOLDER
    size = 100
    models_to_use = [x[2] for x in MODELS]
    datasets = utils.get_sentence_datasets_from_folder(folder, size=size, file_name="ent_train.tsv")

    for n, d in datasets:
        print("{} size {}".format(n, len(d)))

    model_to_domain_to_encodings = encode_with_models(datasets, models_to_use, save_folder)
    print("Model keys: {}".format(model_to_domain_to_encodings.keys()))

    dataset_to_states = encode_with_bioword2vec(datasets, save_folder)
    print("BioWordVec keys: {}".format(dataset_to_states.keys()))

    model_to_domain_to_encodings.update(dataset_to_states)
    print("Model keys: {}".format(model_to_domain_to_encodings.keys()))


if __name__ == "__main__":
    main()
