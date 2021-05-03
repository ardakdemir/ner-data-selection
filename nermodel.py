import torch
import numpy as np
import time
from tqdm import tqdm
import argparse
import json
import h5py
import os
from torch.utils.data import Dataset, DataLoader
from transformers import *
from collections import defaultdict
from itertools import product
import logging
import utils
from gensim.models import FastText, KeyedVectors

from gensim.utils import tokenize


class NerModel(nn.Module):

    def __init__(self, args, model_tuple):
        super(NerModel, self).__init__()

        model_class, tokenizer_class, model_name, save_name = model_tuple
        tokenizer = tokenizer_class.from_pretrained(model_name)

        self.input_dims, self.output_dim = args.input_dim, args.output_dim
        self.model = model_class.from_pretrained(model_name)
        self.classifier = nn.Linear(self.input_dims, self.output_dim)


    def forward(self, bert_input):
        output = model(input_ids)
        last_hidden_states = output[0]
        print("Bert ooutput shape", last_hidden_states.shape)
        class_output = self.classifier(last_hidden_states)
        print("Classifier outtput shape", class_output.shape)
        return class_output
