from transformers import AdamW, RobertaModel, BertForTokenClassification, \
    RobertaForTokenClassification, DistilBertForTokenClassification, \
    RobertaTokenizer, DistilBertModel, DistilBertTokenizer, BertModel, \
    BertTokenizer

from transformers import BertTokenizer, BertForSequenceClassification
import torch


def load_weights_with_skip(model, weights, skip_layers=["bert.pooler", "classifier"]):
    for x in weights:
        if any([x.startswith(layer) for layer in skip_layers]):
            print("Skipping loading {}".format(x))
            continue
        model[x] = weights[x]
    return model


model_save_path = "model_save_deneme.pkh"

tokenclass_model_tuple = (BertForTokenClassification, BertTokenizer, "dmis-lab/biobert-v1.1", "BioBERT")
seqclass_model_tuple = (BertForSequenceClassification, BertTokenizer, "dmis-lab/biobert-v1.1", "BioBERT")

# Define classifier
model_class, tokenizer_class, model_name, save_name = seqclass_model_tuple
tokenizer = tokenizer_class.from_pretrained(model_name)
input_dims, output_dim = 768, 10
model = model_class.from_pretrained(model_name, return_dict=True, num_labels=output_dim)
print("Sequece classifier params")
for param_tensor in model.state_dict():
    print(param_tensor, model.state_dict()[param_tensor].shape)

# Save Sequence-classifier
best_model_weights = model.state_dict()
torch.save(best_model_weights, model_save_path)

# Load classifier as token-classifier
model_class, tokenizer_class, model_name, save_name = tokenclass_model_tuple
input_dims, output_dim = 768, 5
model = model_class.from_pretrained(model_name, return_dict=True, num_labels=output_dim)
load_weights = torch.load(model_save_path)

print("Token classifier params")
for param_tensor in model.state_dict():
    print(param_tensor, model.state_dict()[param_tensor].shape)

print("Loaded params ")
for x in load_weights:
    print(x, load_weights[x].shape)


load_weights_with_skip(model, load_weights)
