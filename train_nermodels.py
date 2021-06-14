import torch
import numpy as np
import time
from tqdm import tqdm
import argparse
import json
import h5py
import os
from transformers import AdamW, RobertaModel, BertForTokenClassification, \
    RobertaForTokenClassification, DistilBertForTokenClassification, \
    RobertaTokenizer, DistilBertModel, DistilBertTokenizer, BertModel, \
    BertTokenizer
from collections import defaultdict
from itertools import product
import logging
from torch.nn import CrossEntropyLoss, MSELoss
from conll_eval import evaluate_conll_file
from ner_dataset import NerDataset, NerDatasetLoader
from nermodel import NerModel
from gensim.utils import tokenize
import matplotlib.pyplot as plt

CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
UNK_TOKEN = "[UNK]"
PAD_TOKEN = "[PAD]"

SPECIAL_TOKENS = [CLS_TOKEN, SEP_TOKEN, UNK_TOKEN, PAD_TOKEN]

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
model_names = ["robertaLarge", "distilbertBaseUncased", "BioBERT", "BioWordVec"]


# dataset_list = ['BC4CHEMD', 'BC2GM', 'BC5CDR', 'conll-eng']


# train_file_path = "/Users/ardaakdemir/bioMLT_folder/biobert_data/datasets/BioNER_2804/BC2GM/ent_train.tsv"
# dev_file_path = "/Users/ardaakdemir/bioMLT_folder/biobert_data/datasets/BioNER_2804/BC2GM/ent_devel.tsv"

def parse_args():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Working  on {}".format(device))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", default="../biobert_data/datasets/BioNER_2804_labeled_cleaned", type=str,
        required=False
    )
    parser.add_argument(
        "--evaluate_root", default="../biobert_data/datasets/BioNER_2804_labeled_cleaned", type=str,
        required=False
    )
    parser.add_argument(
        "--train_file_path", default="../biobert_data/datasets/BioNER_2804_labeled_cleaned/BC2GM/ent_train.tsv",
        type=str,
        required=False
    )

    parser.add_argument(
        "--dev_file_path", default="../biobert_data/datasets/BioNER_2804_labeled_cleaned/BC2GM/ent_devel.tsv", type=str,
        required=False
    )
    parser.add_argument(
        "--test_file_path", default="../biobert_data/datasets/BioNER_2804_labeled_cleaned/BC2GM/ent_test.tsv", type=str,
        required=False
    )
    parser.add_argument(
        "--save_folder", default="../dataselect_lda_nerresult_1306", type=str, required=False,
        help="The path to save everything..."
    )
    parser.add_argument(
        "--dc_model_weight_path", default="../dc_result/best_model_weights.pkh", type=str, required=False,
    )
    parser.add_argument(
        "--dc_save_root", default="../dc_results_2205", type=str, required=False,
    )
    parser.add_argument(
        "--load_dc_model", default=False, action="store_true", help="Whether to load dc model weights..."
    )
    parser.add_argument(
        "--save_folder_root", default="../dataselect_nerresult_2205", type=str, required=False,
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
        "--target_dataset_path", default="../biobert_data/datasets/BioNER_2804/BC2GM", type=str, required=False)
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


def plot_arrays(arrays, names, xlabel, ylabel, save_path):
    plt.figure(figsize=(18, 12))
    for n, a in zip(names, arrays):
        plt.plot(a, label=n)
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path)
    plt.close()


def train(args):
    #     biobert_model_tuple = MODELS[-1]
    # model_tuple = (BertModel, BertTokenizer, "bert-base-uncased", "Bert-base")
    model_tuple = (BertForTokenClassification, BertTokenizer, "dmis-lab/biobert-v1.1", "BioBERT")
    dataset_loaders = {}

    save_folder = args.save_folder
    if not os.path.isdir(save_folder): os.makedirs(save_folder)
    size = args.size
    batch_size = args.batch_size

    target_dataset_path = args.target_dataset_path

    dev_file_path = os.path.join(target_dataset_path, "ent_devel.tsv")
    test_file_path = os.path.join(target_dataset_path, "ent_test.tsv")
    train_file_path = args.train_file_path if not args.dev_only else dev_file_path

    target_dataset = os.path.split(target_dataset_path)[-1]
    train_dataset_name = os.path.split(os.path.split(train_file_path)[0])[-1]

    print("Target dataset: {}\nTrain {} dev {} test {}...\n".format(target_dataset, train_file_path, dev_file_path,
                                                                    test_file_path))
    model_name = model_tuple[2]
    tokenizer = BertTokenizer.from_pretrained(model_name)

    ner_dataset = NerDataset(train_file_path, size=size)
    dataset_loader = NerDatasetLoader(ner_dataset, tokenizer, batch_size=batch_size)
    dataset_loaders["train"] = dataset_loader
    num_classes = len(dataset_loader.dataset.label_vocab)
    args.output_dim = num_classes
    print("Label vocab: {}".format(ner_dataset.label_vocab.w2ind))
    eval_ner_dataset = NerDataset(dev_file_path, size=size)
    eval_ner_dataset.label_vocab = ner_dataset.label_vocab
    eval_ner_dataset.token_vocab = ner_dataset.token_vocab
    eval_dataset_loader = NerDatasetLoader(eval_ner_dataset, tokenizer, batch_size=batch_size)
    dataset_loaders["devel"] = eval_dataset_loader

    test_ner_dataset = NerDataset(test_file_path, size=size)
    test_ner_dataset.label_vocab = ner_dataset.label_vocab
    test_ner_dataset.token_vocab = ner_dataset.token_vocab
    test_dataset_loader = NerDatasetLoader(test_ner_dataset, tokenizer, batch_size=batch_size)
    dataset_loaders["test"] = test_dataset_loader

    model = NerModel(args, model_tuple)
    if args.load_dc_model:
        load_path = args.dc_model_weight_path
        print("Loading model from : {}".format(load_path))
        dc_weights = torch.load(load_path)
        model = load_weights_with_skip(model, dc_weights)
    trained_model, train_result, class_to_idx = train_model(model, dataset_loaders, save_folder, args)

    # Plot train/dev losses
    plot_save_path = os.path.join(save_folder, "loss_plot.png")
    plot_arrays([train_result["train_losses"], train_result["dev_losses"]], ["train", "dev"], "epochs", 'loss',
                plot_save_path)

    # Evaluate on test_set
    save_path = os.path.join(save_folder, "conll_testout.txt")
    test_pre, test_rec, test_f1, test_loss = evaluate(trained_model, dataset_loaders["test"], save_path)

    # Save result
    result_save_path = os.path.join(save_folder, "results.json")
    result = {"model_name": model_name,
              "train_size": len(ner_dataset),
              "target_dataset": target_dataset,
              "precision": test_pre,
              "recall": test_rec,
              "test_loss": test_loss,
              "train_dataset_name": train_dataset_name,
              "f1": test_f1,
              "train_result": train_result}
    with open(result_save_path, "w") as j:
        json.dump(result, j)

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
    pad_index = dataset_loader.dataset.label_vocab.w2ind["[PAD]"]
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
            b, n, c = output.shape
            for l in label:
                labels.extend(l.detach().cpu().tolist())
            pred = torch.argmax(output, dim=2)
            for p in pred:
                preds.extend(p.detach().cpu().tolist())
            for t, p, l in zip(tokens, pred.detach().cpu().tolist(), label.detach().cpu().tolist()):
                conll_data.append(list(zip(t, l, p)))
        # if i > 5: break
    write_to_conll_format(conll_data, dataset_loader.dataset.label_vocab, save_path)
    pre, rec, f1 = evaluate_conll_file(open(save_path).readlines())
    print("Pre: {} Rec: {} F1: {}".format(pre, rec, f1))
    print("{} preds {} labels...".format(len(preds), len(labels)))

    total_loss = total_loss / len(dataset_loader.dataset)
    print("Loss: {}".format(total_loss))
    return pre, rec, f1, total_loss


def train_model(model, dataset_loaders, save_folder, args):
    model_save_path = os.path.join(save_folder, "best_model_weights.pkh")
    eval_save_path = os.path.join(save_folder, "conll_dev_out.txt")

    eval_interval = args.eval_interval if args.eval_interval != -1 else len(dataset_loaders["train"])
    epoch_num = args.epoch_num

    model.to(device)
    model = model.train()

    optimizer = AdamW(model.parameters(), lr=2e-5)
    pad_index = dataset_loaders["train"].dataset.label_vocab.w2ind["[PAD]"]

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
            inputs, label, tokens = train_loader[i]
            inputs = inputs.to(device)
            label = label.to(device)
            output = model(inputs, labels=label)
            loss = output.loss
            logits = output.logits
            # b, n, c = output.shape
            # output = output.view(-1, n)
            # label = label.to(device)
            # loss = criterion(output, label.view(-1))
            total_loss += loss.detach().cpu().item()
            total_num += label.shape[0]
            loss.backward()
            optimizer.step()
            # print("Loss", loss.item())
            if (i + 1) % 100 == 0:
                print("Loss at {}: {}".format(str(i + 1), total_loss / (i + 1)))
        train_loss = total_loss / total_num
        train_losses.append(train_loss)
        train_loader.for_eval = True
        pre, rec, f1, dev_loss = evaluate(model, train_loader, eval_save_path)
        pre, rec, f1, dev_loss = evaluate(model, eval_loader, eval_save_path)
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


def inference_wrapper():
    args = parse_args()
    model_path = args.model_path
    class_dict_path = args.class_dict_path
    save_folder_root = args.save_folder

    assert os.path.exists(class_dict_path) and os.path.exists(
        model_path), "model_path and class_dict_path must exist in inference"
    if args.multiple:
        for d in dataset_list:
            print("Inference for {}".format(d))
            my_save_folder = os.path.join(save_folder_root, d)
            if not os.path.exists(my_save_folder): os.makedirs(my_save_folder)
            args.dev_file_path = os.path.join(args.evaluate_root, d, "ent_devel.tsv")
            args.test_file_path = os.path.join(args.evaluate_root, d, "ent_test.tsv")
            args.save_folder = my_save_folder
            print("Saving {} results to {} ".format(d, my_save_folder))
            inference(model_path, class_dict_path, args)
    else:
        dataset_name = os.path.split(args.evaluate_root)[-1]
        my_save_folder = os.path.join(save_folder_root)
        if not os.path.exists(my_save_folder): os.makedirs(my_save_folder)
        args.dev_file_path = os.path.join(args.evaluate_root, "ent_devel.tsv")
        args.test_file_path = os.path.join(args.evaluate_root, "ent_test.tsv")
        args.save_folder = my_save_folder
        print("Saving {} results to {} ".format(dataset_name, my_save_folder))
        inference(model_path, class_dict_path, args)


def load_weights_with_skip(model, weights, skip_layers=["bert.pooler", "classifier"]):
    my_weights = {}
    for x in weights:
        if any([x.startswith(layer) for layer in skip_layers]):
            print("Skipping loading {}".format(x))
            continue
        # model[x] = weights[x]
        my_weights[x] = weights[x]
    model.load_state_dict(my_weights, strict=False)
    return model


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


def inference(model_path, class_dict_path, args):
    with open(class_dict_path, "rb") as js:
        class_dict = json.load(js)

    save_folder = args.save_folder
    eval_save_path = os.path.join(save_folder, "conll_dev_out.txt")
    test_file_path = args.test_file_path

    model_tuple = (BertForTokenClassification, BertTokenizer, "dmis-lab/biobert-v1.1", "BioBERT")
    model_name = model_tuple[2]
    tokenizer = BertTokenizer.from_pretrained(model_name)
    batch_size = args.batch_size

    eval_ner_dataset = NerDataset(test_file_path)
    eval_ner_dataset.label_vocab.set_w2ind(class_dict)
    test_dataset_loader = NerDatasetLoader(eval_ner_dataset, tokenizer, batch_size=batch_size)

    num_classes = len(class_dict)
    args.output_dim = num_classes

    model = NerModel(args, model_tuple)
    print("Loading weights from {}".format(model_path))
    load_model(model, model_path)
    model.to(device)
    pre, rec, f1, total_loss = evaluate(model, test_dataset_loader, eval_save_path)

    # Save result
    result_save_path = os.path.join(save_folder, "results.json")
    result = {"model_path": model_path,
              "precision": pre,
              "recall": rec,
              "test_loss": total_loss,
              "f1": f1}
    with open(result_save_path, "w") as j:
        json.dump(result, j)


def train_all_datasets(save_folder_root, my_dataset_list, args):
    results = {}
    for d in my_dataset_list:
        my_save_folder = os.path.join(save_folder_root, d)
        args.target_dataset_path = os.path.join(args.dataset_root, d)
        args.train_file_path = os.path.join(args.dataset_root, d, "ent_train.tsv")
        args.dev_file_path = os.path.join(args.evaluate_root, d, "ent_devel.tsv")
        args.test_file_path = os.path.join(args.evaluate_root, d, "ent_test.tsv")
        args.save_folder = my_save_folder
        print("Saving {} results to {} ".format(d, my_save_folder))
        print("Train {} dev {} test {}".format(args.train_file_path, args.dev_file_path, args.test_file_path))
        res = train(args)
        results[d] = res
    return results


def main():
    args = parse_args()
    save_folder_root = args.save_folder
    evaluate_root_allmodels = args.evaluate_root
    dataset_root_allmodels = args.dataset_root
    if args.inference:
        inference_wrapper()
    elif args.multiple:
        if args.multi_model:
            for model in model_names:
                save_folder_allmodels = args.save_folder_root
                save_folder_root = os.path.join(save_folder_allmodels, model)
                args.evaluate_root = os.path.join(evaluate_root_allmodels, model)
                args.dataset_root = os.path.join(dataset_root_allmodels, model)
                train_all_datasets(save_folder_root, dataset_list, args)
        else:
            for d in dataset_list:
                print("Training for {}".format(d))
                my_save_folder = os.path.join(save_folder_root, d)
                args.target_dataset_path = os.path.join(args.dataset_root, d)
                args.train_file_path = os.path.join(args.dataset_root, d, "ent_train.tsv")
                args.dev_file_path = os.path.join(args.evaluate_root, d, "ent_devel.tsv")
                args.test_file_path = os.path.join(args.evaluate_root, d, "ent_test.tsv")
                if args.load_dc_model:
                    args.dc_model_weight_path = os.path.join(args.dc_save_root, d, "best_model_weights.pkh")
                    print("Will load dc model from : {}".format(args.dc_model_weight_path))
                args.save_folder = my_save_folder
                print("Saving {} results to {} ".format(d, my_save_folder))
                print("Train {} dev {} test {}".format(args.train_file_path, args.dev_file_path, args.test_file_path))
                train(args)
    else:
        d = args.dataset_root
        print("Training for {}".format(d))
        my_save_folder = os.path.join(save_folder_root, os.path.split(d)[-1])
        args.train_file_path = os.path.join(d, "ent_train.tsv")
        args.dev_file_path = os.path.join(args.evaluate_root, "ent_devel.tsv")
        args.test_file_path = os.path.join(args.evaluate_root, "ent_test.tsv")
        args.save_folder = my_save_folder
        print("Saving {} results to {} ".format(d, my_save_folder))
        if args.load_dc_model:
            args.dc_model_weight_path = os.path.join(args.dc_save_root, d, "best_model_weights.pkh")
            print("Will load dc model from : {}".format(args.dc_model_weight_path))
        print("Train {} dev {} test {}".format(args.train_file_path, args.dev_file_path, args.test_file_path))
        train(args)


if __name__ == "__main__":
    main()
