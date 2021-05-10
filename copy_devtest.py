import os

file_names = ["ent_devel.tsv","ent_test.tsv"]


src_folder_root = "biobert_data/datasets/BioNER_2804"
save_folder_root = "dataselection_0905"
dataset_list = ['s800', 'NCBI-disease', 'JNLPBA', 'linnaeus', 'BC4CHEMD', 'BC2GM', 'BC5CDR', 'conll-eng']
model_list = ['robertaLarge', 'distilbertBaseUncased', 'BioBERT', 'BioWordVec']
for model in model_list:
    my_folder = os.path.join(save_folder_root,model)