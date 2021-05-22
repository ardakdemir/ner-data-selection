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

SAVE_FOLDER = "../domain_classification_2205"
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


def get_class_dataset(vector_dict, size=None):
    vecs = []
    labels = []
    for k, v in vector_dict.items():
        if size:
            np.random.shuffle(v)
            v = v[:size]
        vecs.extend(v)
        labels.extend([k.split("_")[0]] * len(v))
    return vecs, labels


def split_dataset(vecs, labels, ratio=0.7):
    data = list(zip(vecs, labels))
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


def domain_classify():
    experiment_list = [(False, -1), (True, 50),(True, 100),(True, 200)]
    model_names = ["BioWordVec", "distilbertBaseUncased", "robertaLarge", "BioBERT"]
    num_experiments = 3
    size = 5000
    result_json = {}
    predictions = {}  # Use for confusion matrix
    for model_name in model_names:
        folder = os.path.join(ROOT_FOLDER, model_name)
        if not os.path.isdir(folder):
            continue
        vect_dict = load_vectors_from_folder(folder)
        result_json[model_name] = {}
        best_f1 = 0
        best_preds = None
        for exp in tqdm(experiment_list, desc="Experiment"):
            print("Starting {} {}".format(model_name, exp[-1]))
            pres = []
            recs = []
            f1s = []
            with_pca, pca_dim = exp
            for i in tqdm(range(num_experiments), desc="repeat"):
                clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
                vecs, labels = get_class_dataset(vect_dict, size=size)
                if with_pca:
                    pca = PCA(n_components=pca_dim)
                    pca_vecs = pca.fit_transform(vecs)
                    print("PCA dim {}".format(pca_vecs.shape))
                else:
                    pca_dim = len(vecs[0])
                    pca_vecs = vecs
                train, test = split_dataset(pca_vecs, labels, ratio=0.8)

                tr_x, tr_y = list(zip(*train))
                ts_x, ts_y = list(zip(*test))
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
            print("Results for {} {}: {}".format(model_name, exp, f1s))
            model_result = {"f1s": f1s, "recs": recs, "pres": pres}
            result_json[model_name][pca_dim] = model_result
            predictions[model_name] = best_preds

    if not os.path.isdir(SAVE_FOLDER):
        os.makedirs(SAVE_FOLDER)
    result_config_json = os.path.join(SAVE_FOLDER, "classification_result.json")
    with open(result_config_json, "w") as o:
        json.dump(result_json, o)

    plot_confusion_matrix(predictions, SAVE_FOLDER)


def main():
    domain_classify()


if __name__ == "__main__":
    main()
