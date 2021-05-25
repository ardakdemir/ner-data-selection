import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.svm import SVC
import json as json
from tqdm import tqdm
import os
import h5py
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
# Plot confusion matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from load_sentence_data import load_sentence_data

SAVE_FOLDER = "../domain_classification_2505"
ROOT_FOLDER = "/home/aakdemir/all_encoded_vectors_0305"


def load_vectors(p):
    with h5py.File(p, "r") as h:
        return h["vectors"][:]


def load_vectors_from_folder(folder):
    vect_dict = {}
    for x in os.listdir(folder):
        name = x.split(".")[0]
        p = os.path.join(folder, x)
        vectors = load_vectors(p)
        vect_dict[name] = vectors
    return vect_dict


def get_class_dataset(sentence_data, size=None):
    vecs = []
    labels = []
    sentences = []
    for k, my_data in sentence_data.items():
        if size:
            np.random.shuffle(my_data)
            my_data = my_data[:size]
        labs, vectors, sents = list(zip(*my_data))
        vecs.extend(vectors)
        labels.extend(labs)
        sentences.extend(sents)
    print("{} vecs {} labels {} sentences".format(len(vecs), len(labels), len(sentences)))
    return vecs, labels, sentences


def split_dataset(data, ratio=0.7):
    # data = list(zip(vecs, labels))
    np.random.shuffle(data)
    i = int(len(data) * ratio)
    return data[:i], data[i:]


def plot_confusion_matrix(best_preds, save_folder):
    for model, preds in best_preds.items():
        ts_y, y_pred = preds
        plt.figure(figsize=(14, 10))
        labels = list(set([x for x in ts_y]).union(set([x for x in y_pred])))
        labels = [x.split("_")[0] for x in labels]
        conf_mat = confusion_matrix(ts_y, y_pred, labels=labels)
        df_cm = pd.DataFrame(conf_mat, labels, labels)
        # conf_mat = [[round(float(x)/sum(r),3)*100 for x in r] for r in conf_mat]
        ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 20}, cbar=False, cmap="YlGnBu", fmt='g')  # font size

        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        plt.yticks(rotation=0)

        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder, "confusion_matrix_{}.pdf".format(model)))


def domain_classify(all_sentences_path=None):
    # experiment_list = [(False, -1), (True, 50),(True, 100),(True, 200)]
    # model_names = ["BioWordVec", "distilbertBaseUncased", "robertaLarge", "BioBERT"]
    experiment_list = [(False, -1)]
    model_names = ["BioBERT"]
    num_experiments = 1
    size = 100
    result_json = {}
    predictions = {}  # Use for confusion matrix
    wrong_save_path = "wrong_examples.json"
    wrong_example_dict = {}
    for model_name in model_names:
        final_wrong_examples = []
        sentence_data = load_sentence_data(all_sentences_path, model_name)
        result_json[model_name] = {}
        best_f1 = 0
        best_preds = None
        for exp in tqdm(experiment_list, desc="Experiment"):
            wrong_examples = []
            print("Starting {} {}".format(model_name, exp[-1]))
            pres = []
            recs = []
            f1s = []
            with_pca, pca_dim = exp
            for i in tqdm(range(num_experiments), desc="repeat"):
                clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                vecs, labels, sentences = get_class_dataset(sentence_data, size=size)
                if with_pca:
                    pca = PCA(n_components=pca_dim)
                    pca_vecs = pca.fit_transform(vecs)
                    print("PCA dim {}".format(pca_vecs.shape))
                else:
                    pca_dim = len(vecs[0])
                    pca_vecs = vecs
                data = list(zip(pca_vecs, labels, sentences))
                train, test = split_dataset(data, ratio=0.8)

                tr_x, tr_y, tr_sents = list(zip(*train))
                ts_x, ts_y, ts_sents = list(zip(*test))
                clf.fit(tr_x, tr_y)
                y_pred = clf.predict(ts_x)
                pre, rec, f1, _ = precision_recall_fscore_support(ts_y, y_pred, average='micro')
                acc = accuracy_score(ts_y, y_pred)
                pres.append(np.round(pre, 3))
                recs.append(np.round(rec, 3))
                f1s.append(np.round(f1, 3) * 100)
                if f1 > best_f1:
                    best_f1 = f1
                    best_preds = (ts_y, y_pred)
                    for p, t, s in zip(y_pred, ts_y, ts_sents):
                        if p != t:
                            wrong_examples.append({"sentence": s, "prediction": p, "true_label": t})
                    final_wrong_examples = wrong_examples
            print("Results for {} {}: {}".format(model_name, exp, f1s))
            model_result = {"f1s": f1s, "recs": recs, "pres": pres}
            result_json[model_name][pca_dim] = model_result
            predictions[model_name] = best_preds
            wrong_example_dict[model_name] = final_wrong_examples

    if not os.path.isdir(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    result_config_json = os.path.join(SAVE_FOLDER, "classification_result.json")
    with open(result_config_json, "w") as o:
        json.dump(result_json, o)

    wrong_save_path = os.path.join(SAVE_FOLDER, wrong_save_path)
    with open(wrong_save_path, "w") as o:
        json.dump(wrong_example_dict, o)
    plot_confusion_matrix(predictions, SAVE_FOLDER)


def main():
    path = "../dataselection_1005_labeled/allsentences_pickle.p"
    domain_classify(path)


if __name__ == "__main__":
    main()
