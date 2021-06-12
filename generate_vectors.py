import torch
import numpy as np
import time
from tqdm import tqdm
import argparse
import json
import h5py
import os
from numpy import dot, inner
from numpy.linalg import norm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer, DistilBertModel, DistilBertTokenizer, BertModel, BertTokenizer
from write_selected_sentences import write_selected_sentences
from collections import defaultdict
from itertools import product
from copy_devtest import copy_devtest
from annotate_all_entities import annotate_all_entities
import logging
import utils
import pickle
from gensim.models import FastText, KeyedVectors
from stopwords import english_stopwords

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from gensim.utils import tokenize

MODELS = [(RobertaModel, RobertaTokenizer, 'roberta-large', "robertaLarge"),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased', "distilbertBaseUncased"),
          (BertModel, BertTokenizer, "dmis-lab/biobert-v1.1", "BioBERT")]

train_file_name = "ent_train.tsv"
test_file_name = "ent_test.tsv"

ROOT_FOLDER = "/home/aakdemir/biobert_data/datasets/BioNER_2804"
SAVE_FOLDER = "/home/aakdemir/all_encoded_vectors_0405"
DEV_SAVE_FOLDER = "/home/aakdemir/all_dev_encoded_vectors_0405"
TEST_SAVE_FOLDER = "/home/aakdemir/bioner_testvectors_3005"
SELECTED_SAVE_ROOT = "../dummy_selected_save_root"
COS_SIM_SAMPLE_SIZE = 1000
BioWordVec_FOLDER = "../biobert_data/bio_embedding_extrinsic"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Working  on {}".format(device))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_folder", default="/home/aakdemir/biobert_data/datasets/BioNER_2804_labeled_cleaned", type=str,
        required=False)
    parser.add_argument(
        "--dataset_name", default="random", type=str, required=False)
    parser.add_argument(
        "--save_folder", default="/home/aakdemir/all_encoded_vectors_0606", type=str, required=False)
    parser.add_argument(
        "--dev_save_folder", default="/home/aakdemir/all_dev_encoded_vectors_0606", type=str, required=False)
    parser.add_argument(
        "--test_save_folder", default="/home/aakdemir/bioner_testvectors_0606", type=str, required=False)
    parser.add_argument(
        "--selected_save_root", default="/home/aakdemir/dataselection_0606_labeled", type=str, required=False)
    parser.add_argument(
        "--random", default=False, action="store_true", required=False)
    parser.add_argument(
        "--repeat", default=4, type=int, required=False)
    parser.add_argument(
        "--select_mode", default="size", choices=["size", "similarity"], required=False)
    parser.add_argument(
        "--selection_method", default="cosine_instance", choices=["cosine_instance", "cosine_subset"], required=False)
    parser.add_argument(
        "--select_size", default=100000, type=int, required=False)
    parser.add_argument(
        "--select_thres", default=0.9, type=float, required=False)
    parser.add_argument(
        "--subset_size", default=20, type=int, required=False)
    parser.add_argument(
        "--train_size", default=50000, type=int, required=False)
    parser.add_argument(
        "--dev_size", default=20000, type=int, required=False)
    parser.add_argument(
        "--biowordvec_folder", default="/home/aakdemir/biobert_data/bio_embedding_extrinsic", type=str, required=False)
    parser.add_argument(
        "--word2vec_folder", default="/home/aakdemir/biobert_data/word2Vec/en/en.bin", type=str, required=False)
    parser.add_argument(
        "--cos_sim_sample_size", default=1000, type=int, required=False)
    args = parser.parse_args()
    args.device = device
    return args


def get_w2v_sent_reps(dataset, model, max_pool=False):
    """
    Encodes the lines in a text file using word2vec
    """
    vecs = []
    sents = []
    toks = []
    labs = []
    for data in dataset:
        labels = [d[-1] for d in data]
        tokens = [d[0] for d in data]
        vec, sent_toks = encode_sent_with_w2v(tokens, model, max_pool)
        vecs.append(vec)
        toks.append(tokens)
        sents.append(" ".join(tokens))
        labs.append(labels)
    return np.stack(vecs), sents, toks, labs


def encode_sent_with_w2v(tokens, model, max_pool=False):
    """
    Encodes a sentence as a sum of its corresponding word2vec embeddings.
    """
    MODEL_SIZE = 300
    vecs = []
    for tok in tokens:
        if tok in model:
            vecs.append(model[tok])
    if len(vecs):
        if max_pool:
            pooled = np.max(np.stack(vecs), axis=0)
        else:
            pooled = np.mean(np.stack(vecs), axis=0)
    else:
        pooled = model['unk']
    return pooled, tokens


def get_all_lda_vector_representations(train_datasets, dev_dataset, num_topics=50):
    train_sizes, train_combined_tokens = combine_all_datasets(train_datasets)
    dev_tokens = [" ".join([d[0] for d in data]) for data in dev_dataset]
    dev_size = len(dev_tokens)
    all_sentences = train_combined_tokens + dev_tokens
    vectorizer = TfidfVectorizer(max_df=0.7, stop_words=english_stopwords)
    tf_idf = vectorizer.fit_transform(all_sentences)
    feature_names = vectorizer.get_feature_names()
    print("Found {} tfidf features".format(feature_names))
    train_lda_vectors = {}

    vecs = tf_idf.toarray()
    print("Shape of input to lda : {}".format(vecs.shape))
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=0)
    topic_vectors = lda.fit_transform(vecs)
    print("Shape of lda output: {}".format(topic_vectors.shape))

    s = 0
    for i, size in enumerate(train_sizes):
        dataset_name = train_datasets[i]
        train_lda_vectors[dataset_name] = topic_vectors[s:s + size]
        s = s + size
    dev_lda_vecs = topic_vectors[-dev_size:]
    return feature_names, train_lda_vectors, dev_lda_vecs


def combine_all_datasets(train_datasets):
    train_sizes = []
    train_combined_tokens = []
    for k, dataset in train_datasets:
        tokens = [" ".join([d[0] for d in data]) for data in dataset]
        train_sizes.append(len(tokens))
        print("{} sentences for {}-train...".format(len(tokens), k))
        train_combined_tokens.extend(tokens)
    return train_sizes, train_combined_tokens


def get_lda_sims(select_data, dev_lda_vecs, sample_size=100):
    all_sims = []
    for d in select_data:
        np.random.shuffle(dev_lda_vecs)
        sample_vecs = dev_lda_vecs[:sample_size]
        my_sim = max([cos_similarity(d[1], v) for v in sample_vecs])
        all_sims.append(my_sim)
    return all_sims


def select_with_lda(folder, train_size, dev_size, select_size):
    dataset_list = ['s800', 'NCBI-disease', 'JNLPBA', 'linnaeus', 'BC4CHEMD', 'BC2GM', 'BC5CDR', 'conll-eng']
    train_datasets = utils.get_datasets_from_folder_with_labels(folder,
                                                                size=train_size,  # Use all training data!!!!
                                                                file_name="ent_train.tsv",
                                                                dataset_list=dataset_list)
    dev_datasets = utils.get_datasets_from_folder_with_labels(folder,
                                                              size=dev_size,  # Use all training data!!!!
                                                              file_name="ent_devel.tsv",
                                                              dataset_list=dataset_list)

    # tf_idf, feature_names, vectorizer = get_all_tfidf_vector_representations(train_datasets, dev_datasets)
    for dataset_name, dataset in dev_datasets:
        print("Getting LDA for {}".format(dataset_name))
        feature_names, train_lda_vectors, dev_lda_vecs = get_all_lda_vector_representations(train_datasets,
                                                                                            dataset)
        select_data = []
        for name, dataset in train_datasets:
            lda_vecs = train_lda_vectors[name]
            for data, vec in zip(dataset, lda_vecs):
                tokens = [d[0] for d in data]
                labels = [d[-1] for d in data]
                select_data.append((name, vec, tokens, labels))
        print("{} sentences to select from in total...".format(len(select_data)))
        all_lda_sims = get_lda_sims(select_data, dev_lda_vecs, sample_size=100)
        all_select_data_with_sims = zip(all_sims, select_data)
        all_select_data_with_sims.sort(reverse=True)
        all_select_data_with_sims = all_select_data_with_sims[:select_size]
        sims, selected_data = list(zip(*all_select_data_with_sims))
        print("selected {} data".format(len(selected_data)))
        for d in selected_data:
            print("source ", d[0])
    # vecs = np.array(self.dataset_tfidf_vectors)
    # print("Shape of dataset vectors : {} ".format(vecs.shape))
    # lda = LatentDirichletAllocation(n_components=self.args.lda_topic_num, random_state=0)
    # topic_vectors = lda.fit_transform(vecs)


def encode_with_bioword2vec(datasets, save_folder):
    dataset_to_states = {}

    model = KeyedVectors.load_word2vec_format(BIOWORDVEC_FOLDER, binary=True)
    for dataset_name, dataset in tqdm(datasets, desc="Datasets"):
        begin = time.time()
        vecs, sents, tokens, labels = get_w2v_sent_reps(dataset, model, max_pool=False)
        dataset_to_states[dataset_name] = {"sents": sents,
                                           "tokens": tokens,
                                           "states": vecs,
                                           "labels": labels}
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
            if save_name not in models_to_use:
                print("Skipping {}".format(save_name))
                continue
            # Load pretrained model/tokenizer
            tokenizer = tokenizer_class.from_pretrained(model_name)
            model = model_class.from_pretrained(model_name)
            model.to(DEVICE)
            model_to_states[save_name] = {"sents": [], "tokens": [], "states": [], "labels": []}
            # Encode text
            start = time.time()
            i = 0
            for data in tqdm(dataset, desc="sentences.."):
                labels = [d[-1] for d in data]
                tokens = [d[0] for d in data]
                sentence = " ".join(tokens)
                model_to_states[save_name]['sents'].append(sentence)
                model_to_states[save_name]['labels'].append(labels)
                model_to_states[save_name]['tokens'].append(tokens)
                if i == 0:
                    print(tokens, labels)
                    i += 1
                input_ids = torch.tensor([tokenizer.encode(sentence, add_special_tokens=True,
                                                           truncation=True,
                                                           max_length=128)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.
                input_ids = input_ids.to(DEVICE)
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


def get_domaindev_vectors(folder, size, models_to_use, DEV_SAVE_FOLDER, dataset_list=None):
    """
        Get the vectors for the development sets of each dataset
    :param model_to_domain_to_encodings:
    :param size:
    :return:
    """
    datasets = utils.get_datasets_from_folder_with_labels(folder, size=size,  # Get all dev data for selection!!
                                                          file_name="ent_devel.tsv",
                                                          dataset_list=dataset_list)
    model_to_domain_to_encodings = encode_with_models(datasets, models_to_use, DEV_SAVE_FOLDER)
    if "BioWordVec" in models_to_use:
        dataset_to_states = encode_with_bioword2vec(datasets, DEV_SAVE_FOLDER)
        model_to_domain_to_encodings.update(dataset_to_states)
    return model_to_domain_to_encodings


def get_domaintest_vectors(folder, size, models_to_use, TEST_SAVE_FOLDER, dataset_list=None):
    """
        Get the vectors for the development sets of each dataset
    :param model_to_domain_to_encodings:
    :param size:
    :return:
    """
    datasets = utils.get_datasets_from_folder_with_labels(folder, size=size, file_name="ent_test.tsv",
                                                          dataset_list=dataset_list)
    model_to_domain_to_encodings = encode_with_models(datasets, models_to_use, TEST_SAVE_FOLDER)
    if "BioWordVec" in models_to_use:
        dataset_to_states = encode_with_bioword2vec(datasets, TEST_SAVE_FOLDER)
        model_to_domain_to_encodings.update(dataset_to_states)
    return model_to_domain_to_encodings


def get_domaintrain_vectors(folder, size, models_to_use, save_folder, dataset_list=None):
    datasets = utils.get_datasets_from_folder_with_labels(folder,
                                                          size=size,  # Use all training data!!!!
                                                          file_name="ent_train.tsv",
                                                          dataset_list=dataset_list)

    for n, d in datasets:
        print("{} size {}".format(n, len(d)))

    model_to_domain_to_encodings = encode_with_models(datasets, models_to_use, save_folder)
    print("Model keys: {}".format(model_to_domain_to_encodings.keys()))

    if "BioWordVec" in models_to_use:
        dataset_to_states = encode_with_bioword2vec(datasets, save_folder)
        print("BioWordVec keys: {}".format(dataset_to_states.keys()))
        model_to_domain_to_encodings.update(dataset_to_states)
    print("Model keys: {}".format(model_to_domain_to_encodings.keys()))
    return model_to_domain_to_encodings


def cos_similarity(a, b):
    return inner(a, b) / (norm(a) * norm(b))


def select_data_with_cosine(data_select_data, domain_encodings, args):
    size = args.select_size
    select_mode = args.select_mode
    print("Domain encoding keys: ", domain_encodings.keys())
    domain_vectors = domain_encodings["states"]
    data_sims = []
    for d in tqdm(data_select_data, desc="sentence"):
        np.random.shuffle(domain_vectors)
        sample_vecs = domain_vectors[:COS_SIM_SAMPLE_SIZE]
        my_sim = max([cos_similarity(d[1], v) for v in sample_vecs])
        data_sims.append(my_sim)
    data_with_sims = list(zip(data_sims, data_select_data))
    data_with_sims.sort(key=lambda d: d[0], reverse=True)
    N = 1
    print("Top {} selected data: {}".format(N, data_with_sims[:N]))
    if select_mode == "size":
        sims, data = list(zip(*data_with_sims))
        data = data[:size]
        print("Selected {} instance  with {} as size ".format(len(data), size))
    else:
        print("Selection mode: {}".format(select_mode))
        thres = args.select_thres
        selected = []
        sims, data = list(zip(*data_with_sims))
        for s, d in zip(sims, data):
            if s < thres:
                break
            selected.append(d)
        data = selected
        print("Selected {} instance  with {} as threshold ".format(len(data), thres))
    return data


def get_topN_subsets(data_with_sims, subset_size, N):
    indices = [i for i in range(len(data_with_sims))]
    np.random.shuffle(indices)
    subsets = []
    for x in range(0, len(indices), subset_size):
        my_inds = indices[x:min(x + subset_size, len(indices))]
        my_subset = [data_with_sims[i] for i in my_inds]
        my_sim = np.mean([x[0] for x in my_subset])
        subsets.append([my_sim, my_subset])

    print("Got {} subsets in total...".format(len(subsets)))
    subsets.sort(key=lambda d: d[0], reverse=True)
    subset_sims, subsets = list(zip(*subsets))
    subsets = subsets[:N]

    print("Returning top {} subsets.".format(N))
    instances = []
    for s in subsets:
        instances.extend(s)
    return instances


def select_data_with_cosine_subset(data_select_data, domain_encodings, args):
    size = args.select_size
    subset_size = args.subset_size
    print("Domain encoding keys: ", domain_encodings.keys())
    domain_vectors = domain_encodings["states"]
    data_sims = []
    for d in tqdm(data_select_data, desc="sentence"):
        np.random.shuffle(domain_vectors)
        sample_vecs = domain_vectors[:COS_SIM_SAMPLE_SIZE]
        my_sim = max([cos_similarity(d[1], v) for v in sample_vecs])
        data_sims.append(my_sim)
    data_with_sims = list(zip(data_sims, data_select_data))
    data_with_sims = get_topN_subsets(data_with_sims, subset_size, size)
    sims, data = list(zip(*data_with_sims))
    return data[:size]


def get_dataselect_data(domaintrain_vectors):
    data = []
    print("Get dataselect data select_data_with_cosine_subsetis called")
    for d, vecs in domaintrain_vectors.items():
        print(d, vecs.keys())
        data.extend(
            [(d, s, tokens, labels) for s, tokens, labels in zip(vecs["states"], vecs["tokens"], vecs["labels"])])
    print("{} sentences. First sentence: {}".format(len(data), data[0]))
    return data


def select_data(model_to_domain_to_encodings, domaindev_vectors, size, args):
    selection_method = args.selection_method
    selection_method_map = {"cosine_instance": select_data_with_cosine,
                            "cosine_subset": select_data_with_cosine_subset}
    selected_sentences = {}
    all_sentences = {}
    for model, domain_to_encodings in domaindev_vectors.items():
        print("Running for {}".format(model))
        selected_sentences[model] = {}
        domaintrain_vectors = model_to_domain_to_encodings[model]
        data_select_data = get_dataselect_data(domaintrain_vectors)
        all_sentences[model] = data_select_data
        selected_sentences[model] = {}

        for d, encodings in tqdm(domaindev_vectors[model].items(), desc="Target dataset"):
            beg = time.time()
            print("Selecting data for {} {} method: {}".format(model, d, selection_method))
            selected_data = selection_method_map[selection_method](data_select_data, encodings, args)
            selected_sentences[model][d] = {"selected_data": selected_data,
                                            "all_target_data": encodings}
            end = time.time()
            t = round(end - beg, 3)
            print("Selected {} data in {} seconds ".format(len(selected_data), t))
            # print("Selected sentence0: {}".format(selected_data[0][-1]))
    return selected_sentences, all_sentences


def plot_selected_sentences(selected_sentences, all_sentences):
    pca = PCA(n_components=2)
    all_vectors_combined = []
    for instance in all_sentences:
        all_vectors_combined.append(instance[1])
    all_pca_vecs = pca.fit_transform(all_vectors_combined)
    # for data in [selected_sentences["selected_data"], selected_sentences["all_target_data"]]:


def get_random_data(root_folder, selected_save_folder, name="random", size=None, file_name="ent_train.tsv"):
    selected_sentences = {name: {}}
    all_datasets = []
    for d in os.listdir(root_folder):
        file_path = os.path.join(root_folder, d, file_name)
        dataset = utils.get_tokens_from_dataset_with_labels(file_path, size=size)
        all_datasets.extend(dataset)
    print("{} sentences in total...".format(len(all_datasets)))
    for d in os.listdir(root_folder):
        np.random.shuffle(all_datasets)
        dataset = all_datasets[:size] if size is not None else all_datasets
        selected_data = [list(zip(*sent)) for sent in dataset]
        print("First selected sentence: {}".format(selected_data[0]))
        selected_sentences[name][d] = {"selected_data": selected_data}
    write_selected_sentences(selected_sentences, selected_save_folder, file_name="ent_train.tsv")
    copy_devtest(root_folder, selected_save_folder, model_list=[name])
    selected_save_folder = os.path.join(selected_save_folder, name)
    annotate_all_entities(selected_save_folder, train_file_name, test_file_name)


def select_store_data(models_to_use, dataset_list, args):
    global ROOT_FOLDER
    global DEV_SAVE_FOLDER
    global SAVE_FOLDER
    global BIOWORDVEC_FOLDER
    global COS_SIM_SAMPLE_SIZE
    ROOT_FOLDER = args.root_folder
    DEV_SAVE_FOLDER = args.dev_save_folder
    SAVE_FOLDER = args.save_folder
    BIOWORDVEC_FOLDER = args.biowordvec_folder
    SELECTED_SAVE_ROOT = args.selected_save_root
    COS_SIM_SAMPLE_SIZE = args.cos_sim_sample_size
    select_size = args.select_size
    train_size = args.train_size
    dev_size = args.dev_size
    model_to_domain_to_encodings = get_domaintrain_vectors(ROOT_FOLDER, train_size, models_to_use, SAVE_FOLDER)
    domaindev_vectors = get_domaindev_vectors(ROOT_FOLDER, dev_size, models_to_use, DEV_SAVE_FOLDER, dataset_list)
    print("Domain vector keys : {}".format(domaindev_vectors.keys()))
    selected_sentences, all_sentences = select_data(model_to_domain_to_encodings, domaindev_vectors,
                                                    select_size, args)
    for m, domain_to_sents in selected_sentences.items():
        for d, sents in domain_to_sents.items():
            print("Selected {}/{} sentences using {} target vectors...".format(len(sents["selected_data"]),
                                                                               len(all_sentences[m]),
                                                                               len(sents["all_target_data"]["sents"])))

    if not os.path.exists(SELECTED_SAVE_ROOT):
        os.makedirs(SELECTED_SAVE_ROOT)

    selected_pickle_save_path = os.path.join(SELECTED_SAVE_ROOT, "selected_pickle.p")
    pickle.dump(selected_sentences, open(selected_pickle_save_path, "wb"))

    allsentences_pickle_save_path = os.path.join(SELECTED_SAVE_ROOT, "allsentences_pickle.p")
    pickle.dump(all_sentences, open(allsentences_pickle_save_path, "wb"))

    write_selected_sentences(selected_sentences, SELECTED_SAVE_ROOT, file_name="ent_train.tsv")
    copy_devtest(ROOT_FOLDER, SELECTED_SAVE_ROOT, model_list=models_to_use)


def data_selection_for_all_models():
    args = parse_args()
    global ROOT_FOLDER
    global DEV_SAVE_FOLDER
    global SAVE_FOLDER
    global BIOWORDVEC_FOLDER
    global COS_SIM_SAMPLE_SIZE
    ROOT_FOLDER = args.root_folder
    DEV_SAVE_FOLDER = args.dev_save_folder
    SAVE_FOLDER = args.save_folder
    BIOWORDVEC_FOLDER = args.biowordvec_folder
    SELECTED_SAVE_ROOT = args.selected_save_root
    COS_SIM_SAMPLE_SIZE = args.cos_sim_sample_size
    select_size = args.select_size
    models_to_use = [x[-1] for x in MODELS]
    models_to_use = models_to_use + ["BioWordVec"]

    dataset_list = ['s800', 'NCBI-disease', 'JNLPBA', 'linnaeus', 'BC4CHEMD', 'BC2GM', 'BC5CDR', 'conll-eng']
    # dataset_list = ['s800']

    select_store_data(models_to_use, dataset_list, args)


def save_test_vectors(ROOT_FOLDER, size, models_to_use, TEST_SAVE_FOLDER, dataset_list):
    model_to_domain_to_encodings = get_domaintest_vectors(ROOT_FOLDER, size, models_to_use, TEST_SAVE_FOLDER,
                                                          dataset_list=dataset_list)
    allsentences_pickle_save_path = os.path.join(SELECTED_SAVE_ROOT, "alltest_sentences_pickle.p")
    pickle.dump(model_to_domain_to_encodings, open(allsentences_pickle_save_path, "wb"))


def main():
    args = parse_args()
    global ROOT_FOLDER
    global DEV_SAVE_FOLDER
    global TEST_SAVE_FOLDER
    global SAVE_FOLDER
    global BIOWORDVEC_FOLDER
    global COS_SIM_SAMPLE_SIZE
    ROOT_FOLDER = args.root_folder
    DEV_SAVE_FOLDER = args.dev_save_folder
    SAVE_FOLDER = args.save_folder
    BIOWORDVEC_FOLDER = args.biowordvec_folder
    SELECTED_SAVE_ROOT = args.selected_save_root
    COS_SIM_SAMPLE_SIZE = args.cos_sim_sample_size
    dataset_name = args.dataset_name
    select_size = args.select_size
    # if args.random:
    #     for r in range(args.repeat):
    #         print("Generating random dataset {}".format(r + 1))
    #         dataset_name = "random_{}".format(r)
    #         get_random_data(ROOT_FOLDER, SELECTED_SAVE_ROOT, dataset_name, select_size, file_name="ent_train.tsv")
    # else:
    #     data_selection_for_all_models()
    select_with_lda(ROOT_FOLDER, 1000, 100, 300)
    # TEST_SAVE_FOLDER = args.test_save_folder
    # models_to_use = [x[-1] for x in [MODELS[-1]]]
    # models_to_use = models_to_use + ["BioWordVec"]
    # dataset_list = ['s800', 'NCBI-disease', 'JNLPBA', 'linnaeus', 'BC4CHEMD', 'BC2GM', 'BC5CDR', 'conll-eng']
    # save_test_vectors(ROOT_FOLDER, None, models_to_use, TEST_SAVE_FOLDER, dataset_list)


if __name__ == "__main__":
    main()
