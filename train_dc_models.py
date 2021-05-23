import torch
import numpy as np
import time
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader

import json
import h5py
import os
import json
from transformers import AdamW, RobertaModel, BertForTokenClassification, \
    RobertaForTokenClassification, DistilBertForTokenClassification, \
    RobertaTokenizer, DistilBertModel, DistilBertTokenizer, BertModel, \
    BertTokenizer, BertForSequenceClassification
from collections import defaultdict
from itertools import product
import logging
from torch.nn import CrossEntropyLoss, MSELoss
from conll_eval import evaluate_conll_file
from ner_dataset import Vocab
from nermodel import NerModel
from gensim.utils import tokenize
import matplotlib.pyplot as plt

CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"

SPECIAL_TOKENS = [CLS_TOKEN, SEP_TOKEN, UNK_TOKEN, PAD_TOKEN]
special_tokens = [CLS_TOKEN, SEP_TOKEN, UNK_TOKEN, PAD_TOKEN]

MODELS = [(RobertaModel, RobertaTokenizer, 'roberta-large', "robertaLarge"),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased', "distilbertBaseUncased"),
          (BertModel, BertTokenizer, "dmis-lab/biobert-v1.1", "BioBERT")]

ROOT_FOLDER = "/home/aakdemir/biobert_data/datasets/BioNER_2804"
SAVE_FOLDER = "/home/aakdemir/all_encoded_vectors_0305"
CONLL_SAVE_PATH = "conll_output_0505.txt"
BioWordVec_FOLDER = "../biobert_data/bio_embedding_extrinsic"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_list = ['s800', 'NCBI-disease', 'JNLPBA', 'linnaeus', 'BC4CHEMD', 'BC2GM', 'BC5CDR', 'conll-eng']

# model_names = ["random_0","random_1","random_2","random_3"]
model_names = ["BioBERT"]


# dataset_list = ['BC4CHEMD', 'BC2GM', 'BC5CDR', 'conll-eng']


# train_file_path = "/Users/ardaakdemir/bioMLT_folder/biobert_data/datasets/BioNER_2804/BC2GM/ent_train.tsv"
# dev_file_path = "/Users/ardaakdemir/bioMLT_folder/biobert_data/datasets/BioNER_2804/BC2GM/ent_devel.tsv"


def read_dc_dataset(file_path):
    dataset = json.load(open(file_path, "r"))
    sentences, labels = dataset["sentences"], dataset["labels"]
    return sentences, labels


def get_vocab(tokens):
    w2ind = {}
    for s in special_tokens:
        w2ind[s] = len(w2ind)
    for sent in tokens:
        for tok in sent:
            if tok not in w2ind:
                w2ind[tok] = len(w2ind)
    return w2ind


def get_label_vocab(labels):
    w2ind = {}
    for label in labels:
        if label not in w2ind:
            w2ind[label] = len(w2ind)
    return w2ind


class DCDataset(Dataset):
    def __init__(self, file_path, size=None):
        sentences, labels = read_dc_dataset(file_path)
        token_vocab, label_vocab = get_vocab(sentences), get_label_vocab(labels)
        self.token_vocab = Vocab(token_vocab)
        self.label_vocab = Vocab(label_vocab)
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        sentence = self.sentences[index]
        label = self.labels[index]
        return sentence, label


class DCDatasetLoader:
    def __init__(self, dataset, tokenizer, batch_size=5, for_eval=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.for_eval = for_eval

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)

    def __getitem__(self, index):
        inps = []
        labs = []
        raw_sentences = []
        final_labels = []
        if self.for_eval:
            index = index * self.batch_size
        else:
            index = np.random.randint(0, len(self.dataset) / self.batch_size)
        for b in range(self.batch_size):
            index = (index + b) % len(self.dataset)
            sentence, label = self.dataset[index]
            inps.append(sentence)
            labs.append(self.dataset.label_vocab.map([label])[0])
            raw_sentences.append(sentence)

        inputs = self.tokenizer(inps, return_tensors="pt", padding=True, truncation=True,
                                max_length=512)
        return inputs, torch.tensor(labs), raw_sentences


def parse_args():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Working  on {}".format(device))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", default="../biobert_data/datasets/BioNER_2505_DC_datasets/BC2GM", type=str,
        required=False
    )
    parser.add_argument(
        "--evaluate_root", default="../biobert_data/datasets/BioNER_2505_DC_datasets/BC2GM", type=str,
        required=False
    )
    parser.add_argument(
        "--train_file_path", default="../biobert_data/datasets/BioNER_2505_DC_datasets/BC2GM/train.json",
        type=str,
        required=False
    )
    parser.add_argument(
        "--test_file_path", default="../biobert_data/datasets/BioNER_2505_DC_datasets/BC2GM/test.json", type=str,
        required=False
    )
    parser.add_argument(
        "--save_folder", default="../dc_result", type=str, required=False,
        help="The path to save everything..."
    )
    parser.add_argument(
        "--save_folder_root", default="../dc_result", type=str, required=False,
        help="The path to save everything..."
    )
    parser.add_argument(
        "--model_path", default="../dataselect_nerresult_0505/best_model_weights.pkh", type=str, required=False,
        help="The path to save everything..."
    )
    parser.add_argument(
        "--class_dict_path", default="../dataselect_nerresult_0505/class_to_idx.json", type=str, required=False,
        help="The path to save everything..."
    )
    parser.add_argument(
        "--target_dataset_path", default="../biobert_data/datasets/BioNER_2505_DC_datasets/BC2GM", type=str,
        required=False)
    parser.add_argument(
        "--input_dims", default=768, type=int, required=False,
    )
    parser.add_argument(
        "--size", default=-1, type=int, required=False,
    )
    parser.add_argument(
        "--eval_interval", default=-1, type=int, required=False,
    )
    parser.add_argument(
        "--epoch_num", default=4, type=int, required=False,
    )
    parser.add_argument(
        "--output_dim", default=6, type=int, required=False,
    )
    parser.add_argument(
        "--batch_size", default=12, type=int, required=False,
    )
    parser.add_argument(
        "--multiple", default=False, action="store_true", help="Run for all datasets"
    )
    parser.add_argument(
        "--multi_model", default=False, action="store_true", help="Run for all model"
    )
    parser.add_argument(
        "--inference", default=False, action="store_true", help="Run inference only."
    )
    parser.add_argument(
        "--dev_only", default=False, action="store_true", help="If True, only uses the dev split for training"
    )
    args = parser.parse_args()
    args.device = device
    return args


def train(args):
    #     biobert_model_tuple = MODELS[-1]
    # model_tuple = (BertModel, BertTokenizer, "bert-base-uncased", "Bert-base")
    model_tuple = (BertForSequenceClassification, BertTokenizer, "dmis-lab/biobert-v1.1", "BioBERT")
    dataset_loaders = {}

    save_folder = args.save_folder
    if not os.path.isdir(save_folder): os.makedirs(save_folder)
    size = args.size
    batch_size = args.batch_size

    target_dataset_path = args.target_dataset_path

    test_file_path = args.test_file_path
    train_file_path = args.train_file_path

    target_dataset = os.path.split(target_dataset_path)[-1]
    train_dataset_name = os.path.split(os.path.split(train_file_path)[0])[-1]

    print("Target dataset: {}\nTrain {} test {}...\n".format(target_dataset, train_file_path, test_file_path))
    model_name = model_tuple[2]
    tokenizer = BertTokenizer.from_pretrained(model_name)

    dc_dataset = DCDataset(train_file_path, size=size)
    dataset_loader = DCDatasetLoader(dc_dataset, tokenizer, batch_size=batch_size)
    dataset_loaders["train"] = dataset_loader
    num_classes = len(dataset_loader.dataset.label_vocab)
    args.output_dim = num_classes
    print("Label vocab: {}".format(dc_dataset.label_vocab.w2ind))

    test_dc_dataset = DCDataset(test_file_path, size=size)
    test_dc_dataset.label_vocab = dc_dataset.label_vocab
    test_dc_dataset.token_vocab = dc_dataset.token_vocab
    test_dataset_loader = DCDatasetLoader(test_dc_dataset, tokenizer, batch_size=batch_size)
    dataset_loaders["test"] = test_dataset_loader

    model = BertForSequenceClassification.from_pretrained(model_name, return_dict=True, num_labels=2)
    trained_model, train_result, class_to_idx = train_model(model, dataset_loaders, save_folder, args)

    # Plot train/dev losses
    plot_save_path = os.path.join(save_folder, "loss_plot.png")
    plot_arrays([train_result["train_losses"], train_result["dev_losses"]], ["train", "dev"], "epochs", 'loss',
                plot_save_path)

    class_to_idx_path = os.path.join(save_folder, "class_to_idx.json")
    with open(class_to_idx_path, "w") as j:
        json.dump(class_to_idx, j)
    return result


def write_to_conll_format(conll_data, label_vocab, save_path):
    s = ""
    map_to_conll_label = lambda x: label_vocab.ind2w[x] if label_vocab.ind2w[x] not in SPECIAL_TOKENS else "O"
    for sent in conll_data:
        n = 0
        for t, l, p in sent:
            n += 1
            if t[:2] == "##": continue
            if t in ["[CLS]", "[SEP]"]: continue
            if t == "[PAD]": break
            # print(sent,i,t)
            i = n
            while sent[i][0][:2] == "##":
                t += sent[i][0][2:]
                i += 1

            s += "{}\t{}\t{}\n".format(t, map_to_conll_label(l), map_to_conll_label(p))
        s += "\n"
    with open(save_path, "w") as o:
        o.write(s)


def evaluate(model, dataset_loader, save_path):
    model = model.eval()
    preds = labels = []
    conll_data = []
    total_loss = 0
    for i in tqdm(range(len(dataset_loader)), desc="evaluation"):
        with torch.no_grad():
            inputs, label, tokens = dataset_loader[i]
            inputs = inputs.to(device)
            label = label.to(device)
            output = model(inputs, labels=label)
            loss = output.loss
            output = output.logits
            total_loss += loss.detach().cpu().item()
            labels.extend(label.detach().cpu().tolist())
            pred = torch.argmax(output, dim=1)
            preds.extend(pred.detach().cpu().tolist())

    pre, rec, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    print("Pre: {} Rec: {} F1: {}".format(pre, rec, f1))
    print("{} preds {} labels...".format(len(preds), len(labels)))

    total_loss = total_loss / len(dataset_loader.dataset)
    print("Loss: {}".format(total_loss))
    return pre, rec, f1, total_loss


def train_model(model, dataset_loaders, save_folder, args):
    model_save_path = os.path.join(save_folder, "best_model_weights.pkh")

    eval_interval = args.eval_interval if args.eval_interval != -1 else len(dataset_loaders["train"])
    epoch_num = args.epoch_num

    model.to(device)
    model = model.train()

    optimizer = AdamW(model.parameters(), lr=2e-5)

    train_losses = []
    dev_losses = []
    dev_f1s = []
    train_loader = dataset_loaders["train"]
    eval_loader = dataset_loaders["test"]
    best_f1 = -1
    best_model = 0
    for j in tqdm(range(epoch_num), desc="Epochs"):
        model = model.train()
        total_loss = 0
        total_num = 0
        train_loader.for_eval = False
        for i in tqdm(range(eval_interval), desc="training"):
            optimizer.zero_grad()
            inputs, labels, sentences = train_loader[i]
            print("Inputs: {} Labels: {}".format(inputs, labels))
            inputs = inputs.to(device)
            labels = labels.to(device)
            output = model(**inputs, labels=labels)
            loss = output.loss
            logits = output.logits
            # b, n, c = output.shape
            # output = output.view(-1, n)
            # label = label.to(device)
            # loss = criterion(output, label.view(-1))
            total_loss += loss.detach().cpu().item()
            total_num += labels.shape[0]
            loss.backward()
            optimizer.step()
            # print("Loss", loss.item())
            if (i + 1) % 100 == 0:
                print("Loss at {}: {}".format(str(i + 1), total_loss / (i + 1)))
        train_loss = total_loss / total_num
        train_losses.append(train_loss)
        train_loader.for_eval = True
        pre, rec, f1, dev_loss = evaluate(model, eval_loader, eval_save_path)
        print("\n==== Result for epoch {} === F1: {} ====\n".format(j + 1, f1))
        dev_f1s.append(f1)
        dev_losses.append(dev_loss)
        if f1 > best_f1:
            best_f1 = f1
            best_model_weights = model.state_dict()
            torch.save(best_model_weights, model_save_path)

    model.load_state_dict(torch.load(model_save_path))
    return model, {"train_losses": train_losses,
                   "dev_losses": dev_losses,
                   "dev_f1s": dev_f1s,
                   "class_to_idx": dataset_loaders["train"].dataset.label_vocab.w2ind,
                   "best_f1": best_f1}, dataset_loaders["train"].dataset.label_vocab.w2ind


def load_model(model, model_load_path):
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        if param_tensor == "model.classifier.weight":
            print("Before load ", model.state_dict()[param_tensor])
    load_weights = torch.load(model_load_path)
    model.load_state_dict(load_weights)
    for param_tensor in model.state_dict():
        if param_tensor == "model.classifier.weight":
            print("After load ", model.state_dict()[param_tensor])
    return model


def traindc_all_datasets(save_folder_root, my_dataset_list, args):
    results = {}
    for d in my_dataset_list:
        my_save_folder = os.path.join(save_folder_root, d)
        args.target_dataset_path = os.path.join(args.dataset_root, d)
        args.train_file_path = os.path.join(args.dataset_root, d, "ent_train.tsv")
        args.dev_file_path = os.path.join(args.evaluate_root, d, "ent_devel.tsv")
        args.test_file_path = os.path.join(args.evaluate_root, d, "ent_test.tsv")
        args.save_folder = my_save_folder
        print("Saving {} results to {} ".format(d, my_save_folder))
        print(
            "Train {} dev {} test {}".format(args.train_file_path, args.dev_file_path, args.test_file_path))
        res = train(args)
        results[d] = res
    return results


def main():
    args = parse_args()
    save_folder_root = args.save_folder
    if args.inference:
        inference_wrapper()
    elif args.multiple:
        for d in dataset_list:
            print("Training for {}".format(d))
            my_save_folder = os.path.join(save_folder_root, d)
            args.target_dataset_path = os.path.join(args.dataset_root, d)
            args.train_file_path = os.path.join(args.dataset_root, d, "train.json")
            args.test_file_path = os.path.join(args.evaluate_root, d, "test.json")
            args.save_folder = my_save_folder
            print("Saving {} results to {} ".format(d, my_save_folder))
            print("Train {}  test {}".format(args.train_file_path, args.test_file_path))
            train(args)
    else:
        d = args.dataset_root
        print("Training for {}".format(d))
        my_save_folder = os.path.join(save_folder_root, os.path.split(d)[-1])
        args.train_file_path = os.path.join(args.dataset_root, "train.json")
        args.test_file_path = os.path.join(args.evaluate_root, "test.json")
        args.save_folder = my_save_folder
        print("Saving {} results to {} ".format(d, my_save_folder))
        print("Train {} test {}".format(args.train_file_path, args.test_file_path))
        train(args)


if __name__ == "__main__":
    main()
