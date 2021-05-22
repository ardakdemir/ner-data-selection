import os
import subprocess
import sys
from collections import defaultdict, Counter

dataset_list = ['s800', 'NCBI-disease', 'JNLPBA', 'linnaeus', 'BC4CHEMD', 'BC2GM', 'BC5CDR', 'conll-eng']
# model_list = ['robertaLarge', 'distilbertBaseUncased', 'BioBERT', 'BioWordVec']
model_list = ['BioBERT', 'BioWordVec']
file_names = ["ent_train.tsv", "ent_devel.tsv", "ent_test.tsv"]


def convert_to_tsv_example(sentence, labels):
    return "\n".join(["\t".join([s, l]) for s, l in zip(sentence, labels)])


def annotate_sentence(entity_dict, sentence, labels, ent_label="Entity"):
    """
        Given a dict of entities update the labels of
    """
    my_sentence = " ".join(sentence)
    count = 0
    for entity, lab in entity_dict.items():
        lab = ent_label
        if " " + entity + " " in " " + my_sentence + " ":
            words = entity.split(" ")
            word_one = words[0]
            index = sentence.index(word_one)
            if all([label == "O" for label in labels[index:index + len(words)]]):
                labels[index] = "B-" + lab
                for x in range(index + 1, index + len(words)):
                    labels[x] = "I-" + lab
                count = count + 1
    return sentence, labels, count


def search_unannotated_entity_in_sentence(entity, sentence, labels):
    """
        True if found and not annotated else False
    """

    sent = " ".join(sentence)
    if " " + entity + " " in " " + " ".join(sentence) + " ":
        words = entity.split(" ")
        word_one = words[0]
        try:
            index = sentence.index(word_one)
        except:
            print("Entity: {} Sentence: {}".format(entity, sentence))
        if any([label == "O" for label in labels[index:index + len(words)]]):
            return True
        else:
            return False
    return False


def get_sentence_entities(sentence, labels):
    entities = []
    prev_label = "O"
    curr_entity = ""
    for s, l in zip(sentence, labels):
        if l == "O":
            if len(curr_entity) > 0:
                entities.append(curr_entity)
                curr_entity = ""
                prev_label = "O"
            continue
        if l.startswith("B-"):
            if len(curr_entity) > 0:
                entities.append(curr_entity)
                curr_entity = ""
                prev_label = l
            curr_entity = s
        else:
            curr_entity = curr_entity + " " + s
            prev_label = l
    if len(curr_entity) > 0:
        entities.append(curr_entity)
    return entities


def annotate_dataset(entity_dict, file_path, save_path):
    dataset = open(file_path).read().split("\n\n")
    sentences = [[x.split()[0] for x in sent.split("\n") if len(x.split()) > 0] for sent in dataset if
                 len(sent.split("\n")) > 0]
    labels = [[x.split()[-1] for x in sent.split("\n") if len(x.split()) > 0] for sent in dataset if
              len(sent.split("\n")) > 0]
    new_sentences, new_labels = [], []
    for sent, label in zip(sentences, labels):
        new_sent, new_label, _ = annotate_sentence(entity_dict, sent, label, ent_label="Entity")
        new_sentences.append(new_sent)
        new_labels.append(new_label)
    new_dataset = "\n\n".join([convert_to_tsv_example(s, l) for s, l in zip(new_sentences, new_labels)])
    with open(save_path, "w") as o:
        o.write(new_dataset)


def get_entities_from_folder(data_folder, train_file_name, test_file_name):
    oov_rates = {}
    datasets = os.listdir(data_folder)
    all_train_entities = {}
    all_test_entities = {}
    for d in datasets:
        print(d)
        folder = os.path.join(data_folder, d)
        if not os.path.isdir(folder) or train_file_name not in os.listdir(folder):
            continue
        entity_type = d
        test_data_path = "{}/{}".format(folder, test_file_name)
        test_entities = get_entities_from_tsv_dataset(test_data_path)
        train_data_path = "{}/{}".format(folder, train_file_name)
        train_entities = get_entities_from_tsv_dataset(train_data_path)
        #         oov_rate, oovs,found_entities = get_diff_ratio(test_entities,train_entities)
        #         oov_rates[entity_type] = oov_rate
        all_train_entities[entity_type] = train_entities
        all_test_entities[entity_type] = test_entities

    return all_train_entities, all_test_entities, oov_rates


def get_entities_from_tsv_dataset(file_path, tag_type="BIO"):
    sentences = open(file_path).read().split("\n\n")[:-1]
    entities = defaultdict(Counter)
    occur_limit = 2
    if tag_type == "BIO":
        for sent in sentences:
            prev_tag = "O"
            curr_entity = ""
            for token in sent.split("\n"):
                if len(token.split()) > 1:
                    word = token.split()[0]
                    label = token.split()[-1]
                else:
                    word = ""
                    label = "O"
                if label != "O":
                    if label[0] == "I":
                        curr_entity = curr_entity + " " + word
                    elif label[0] == "B":
                        if prev_tag != "O":
                            if len(curr_entity) > 0:
                                entities[prev_tag][curr_entity] += 1
                        prev_tag = label[2:]
                        curr_entity = word
                else:
                    if prev_tag != "O":
                        entities[prev_tag][curr_entity] += 1
                    prev_tag = "O"
                    curr_entity = ""
            if len(curr_entity) > 0 and prev_tag != "O":
                entities[prev_tag][curr_entity] += 1
    pruned_entities = defaultdict(Counter)
    for tag in entities:
        for ent in entities[tag]:
            if entities[tag][ent] >= occur_limit:
                pruned_entities[tag][ent] = entities[tag][ent]
    for tag in pruned_entities:
        print("{} entities occurring >= {} times".format(len(pruned_entities[tag]), occur_limit))
    return pruned_entities

def annotate_all_entities(data_folder, train_file_name, test_file_name,global_label=False):
    all_train_entities, all_test_entities, oov_rates = get_entities_from_folder(data_folder,
                                                                                train_file_name,
                                                                                test_file_name)
    print(len(all_train_entities["conll-eng"]["Entity"]), len(all_test_entities["conll-eng"]["Entity"]))
    for d in all_train_entities.keys():
        all_train_entities[d]["Entity"].update(all_test_entities[d]["Entity"])
    print(len(all_train_entities["conll-eng"]["Entity"]), len(all_test_entities["conll-eng"]["Entity"]))
    all_entities = {}
    for d,ents in all_train_entities.items():
        all_entities.update(ents["Entity"])
    print("Found {} entities in total for annotation".format(len(all_entities)))
    for d in dataset_list:
        print("Annotating  {}".format(d))
        for file in file_names:
            file_path = os.path.join(data_folder, d, file)
            save_path = os.path.join(os.path.split(file_path)[0], "labeled_" + file)
            print("Saving to {}".format(save_path))
            my_entities = all_train_entities[d]["Entity"] if not global_label else all_entities
            print("Number of entities: ",len(my_entities))
            annotate_dataset(my_entities, file_path, save_path)
            cmd = "mv {} {}".format(save_path, file_path)
            subprocess.call(cmd, shell=True)


def annotate_per_model(data_folder):
    train_file_name, test_file_name = "ent_train.tsv", "ent_test.tsv"
    for model in model_list:
        my_data_folder = os.path.join(data_folder, model)
        annotate_all_entities(my_data_folder, train_file_name, test_file_name)

def annotate_root_folder(data_folder):
    train_file_name, test_file_name = "ent_train.tsv", "ent_test.tsv"
    annotate_all_entities(data_folder, train_file_name, test_file_name,True)



def main():
    data_folder = sys.argv[1]
    train_file_name, test_file_name = "ent_train.tsv", "ent_test.tsv"
    annotate_root_folder(data_folder)



if __name__ == "__main__":
    main()
