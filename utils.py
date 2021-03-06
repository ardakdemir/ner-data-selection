import os
import sys
import matplotlib.pyplot as plt
import numpy as np


def plot_arrays(arrays, save_path, x_title=None, y_title=None, names=None, title=None):
    plt.figure(figsize=(12, 8))
    if title:
        plt.title("Loss curve")
    if names:
        for a, n in zip(arrays, names):
            plt.plot(a, label=n)
    else:
        for a in arrays:
            plt.plot(a)
    if names:
        plt.legend()
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.savefig(save_path)


def combine_dictionaries(dics):
    combined = dics[0]
    for dic in dics[1:]:
        for k in combined.keys():
            if k in dic:
                combined[k].update(dic[k])
    return combined


def sort_dataset(dataset, desc=True, sort=True):
    idx = [i for i in range(len(dataset))]
    if not sort:
        return dataset, idx
    zipped = list(zip(dataset, idx))
    zipped.sort(key=lambda x: len(x[0]))
    if desc:
        zipped.reverse()
    dataset, orig_idx = list(zip(*zipped))
    return dataset, orig_idx


def conll_writer(file_name, content, field_names, task_name, verbose=False):
    out = open(file_name, 'w', encoding='utf-8')
    task_dict = dicts[task_name]
    if verbose:
        out.write("{}\n".format("\t".join([k for k in task_dict])))
    init = ["-" for i in range(len(task_dict))]
    for sent in content:
        for id, tok in enumerate(sent):
            for i, f in enumerate(field_names):
                init[task_dict[f]] = tok[i]
                if type(tok[i]) == int:
                    init[task_dict[f]] = str(tok[i])
            if task_name == 'dep':
                init[0] = str(id + 1)
            out.write("{}\n".format("\t".join(init)))
        out.write("\n")
    out.close()


def unsort_dataset(dataset, orig_idx):
    zipped = list(zip(dataset, orig_idx))
    zipped.sort(key=lambda x: x[1])
    dataset, _ = list(zip(*(zipped)))
    return dataset


def get_sentences_from_dataset(file_path, size=None):
    sentences = open(file_path).read().split("\n\n")
    sentences = [" ".join([x.split()[0] for x in sent.split("\n") if len(x.split()) > 0]) for sent in sentences]
    if not size:
        return sentences
    else:
        np.random.shuffle(sentences)
        return sentences[:size]


def get_tokens_from_dataset_with_labels(file_path, size=None,shuffle=False):
    sentences = open(file_path).read().split("\n\n")
    prune_short_sentences_limit = 6
    sentences = [[(x.split()[0], x.split()[-1]) for x in sent.split("\n") if len(x.split()) > 0] for sent in sentences
                 if len(sent.split("\n")) > prune_short_sentences_limit]
    if shuffle:
        np.random.shuffle(sentences)
    if not size:
        return sentences
    else:
        return sentences[:size]


def read_ner_dataset(file_path, size=None):
    dataset = open(file_path).read().split("\n\n")
    sentences = [" ".join([x.split()[0] for x in sent.split("\n") if len(x.split()) > 0]) for sent in dataset]
    labels = [[x.split()[-1] for x in sent.split("\n") if len(sent.split("\n")) > 0] for sent in dataset]
    data = list(zip(sentences, labels))
    if not size:
        return list(zip(*data))
    else:
        np.random.shuffle(data)
        return list(zip(*data[:size]))


def get_sentence_datasets_from_folder(folder, size=None, file_name="ent_train.tsv"):
    return [(f, get_sentences_from_dataset(os.path.join(folder, f, file_name), size=size)) for f in os.listdir(folder)]


def get_datasets_from_folder_with_labels(folder,
                                         size=None,
                                         file_name="ent_train.tsv",
                                         dataset_list=None,
                                         shuffle=False):
    return [(f, get_tokens_from_dataset_with_labels(os.path.join(folder, f, file_name), size=size,shuffle=shuffle)) for f in
            os.listdir(folder) if (dataset_list is None or f in dataset_list)]



def addbiotags(file_name, pref="ENT"):
    d, f = os.path.split(file_name)
    save_path = os.path.join(d, "ent_{}".format(f))
    i_f = open(file_name).readlines()
    s = "\n".join(["" if x.strip() == "" else "{}\t{}".format(x.split()[0],
                                                              x.split()[1] if x.split()[1] == "O" else "{}-{}".format(
                                                                  x.split()[1], pref)) for x in i_f])
    with open(save_path, "w") as o_f:
        o_f.write(s)


if __name__ == "__main__":
    args = sys.argv
    addbiotags(args[1])
