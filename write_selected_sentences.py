import os

selected_sentences = {
    "bioBERT": {"BC2GM": {"selected_data": [("BC4CHEMD", [1, 2, 3], ['I', 'am', 'the', 'first', 'Menieres', 'Disease'],
                                             ['O', 'O', 'O', 'O', 'B-Entity', 'I-Entity']),
                                            ("BC4CHEMD", [1, 2, 3], ['I', 'am', 'the', 'second', 'CANCER', 'DISEASE'],
                                             ['O', 'O', 'O', 'O', 'B-Entity', 'I-Entity'])]},
                "BC4CHEMD": {
                    "selected_data": [("BC4CHEMD", [1, 2, 3], ['I', 'am', 'the', 'first', 'Chemical', 'sentence'],
                                       ['O', 'O', 'O', 'O', 'B-Entity', "O"]),
                                      ("linnaeus", [1, 2, 3], ['I', 'am', 'the', 'second', 's800', 'sentence'],
                                       ['O', 'O', 'O', 'O', 'O', "O"]),
                                      ("s800", [1, 2, 3], ['I', 'am', 'the', 'third', 's800', 'sentence'],
                                       ['O', 'O', 'O', 'O', 'B-Entity', "O"])]}},
    "bioWordVec": {"linnaeus": {"selected_data": [("a", [1, 2, 3], ['I', 'am', 'the', 'first', 'linnaeus', 'sentence'],
                                                   ['O', 'O', 'O', 'O', 'O', "O"]),
                                                  ("a", [1, 2, 3],
                                                   ['I', 'am', 'the', 'second', 'biowordVec', 'sentence'],
                                                   ['O', 'O', 'O', 'O', 'B-Entity', "O"])]}}}

save_root = "../dummy_selected_data_root"


def write_selected_sentences(selected_sentences, save_root, file_name="ent_train.tsv"):
    for model, domains_to_sents in selected_sentences.items():
        for d, data in domains_to_sents.items():
            sents = data["selected_data"]
            save_folder = os.path.join(save_root, model, d)
            print("Save folder", save_folder)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            print("First sentence", sents[0])
            s = "\n\n".join(["\n".join(["{}\t{}".format(t, l) for t, l in zip(sent[-2], sent[-1])]) for sent in sents])
            with open(os.path.join(save_folder, file_name), "w") as o:
                o.write(s)


if __name__ == "__main__":
    write_selected_sentences(selected_sentences, save_root, file_name="ent_train")
