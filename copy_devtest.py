import os
import subprocess
file_names = ["ent_devel.tsv","ent_test.tsv"]


src_folder_root = "biobert_data/datasets/BioNER_2804"
save_folder_root = "dataselection_0905"
dataset_list = ['s800', 'NCBI-disease', 'JNLPBA', 'linnaeus', 'BC4CHEMD', 'BC2GM', 'BC5CDR', 'conll-eng']
model_list = ['robertaLarge', 'distilbertBaseUncased', 'BioBERT', 'BioWordVec']
for model in model_list:
    my_folder = os.path.join(save_folder_root,model)
    for d in dataset_list:
        src = os.path.join(src_folder_root,d)
        dest = os.path.join(my_folder,d)
        if not os.path.isdir(dest):os.makedirs(dest)
        for f in file_names:
            s = os.path.join(src,f)
            d = os.path.join(dest,f)
            cmd = "cp {} {}".format(s,d)
            subprocess.call(cmd,shell=True)
