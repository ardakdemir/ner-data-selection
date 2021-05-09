import torch
import numpy as np
import time
from tqdm import tqdm
import argparse
import torch.nn as nn
import json
import h5py
import os
from transformers import *
from itertools import product


class NerModel(nn.Module):

    def __init__(self, args, model_tuple):
        super(NerModel, self).__init__()

        model_class, tokenizer_class, model_name, save_name = model_tuple
        tokenizer = tokenizer_class.from_pretrained(model_name)

        self.input_dims, self.output_dim = args.input_dims, args.output_dim
        self.model = model_class.from_pretrained(model_name, num_labels=self.output_dim)
        # self.classifier = nn.Linear(self.input_dims, self.output_dim)

    def forward(self, bert_input, labels=None):
        if labels:
            output = self.model(**bert_input)
        else:
            output = self.model(**bert_input)
        return output
