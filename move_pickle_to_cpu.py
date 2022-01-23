import pickle as pkl
from collections import defaultdict
import torch
import os

p = "../bioMLT_folder/entity_vectors_2201/entity_vectors_5000_entities.pkl"
save_path = os.path.join(os.path.split(p)[0],os.path.split(p)[1].split(".")[0]+"_cpu.pkl")
print("Saving to {}".format(save_path))

entity_dict = pkl.load(open(p, "rb"))

new_dict = {}

for k, d in entity_dict.items():
    new_dict[k] = {}
    print(k)
    for name, data in d.items():
        print(name)
        new_dict[k][name] = defaultdict(list)
        for ent,vecs in data.items():
            new_dict[k][name][ent] = torch.mean(torch.stack(vecs),dim=0).detach().cpu().numpy()

with open(save_path, "wb") as p:
    pkl.dump(entity_vector_dict, p)
