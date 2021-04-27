from collections import Counter
import glob
import torch
import numpy as np
import logging
from transformers import BertTokenizer
# from parser.parsereader import bert2token, pad_trunc_batch
from vocab import Vocab, VOCAB_PREF
from utils import sort_dataset, unsort_dataset
from data_cleaner import data_reader

PAD = "[PAD]"
START_TAG = "[CLS]"
END_TAG = "[SEP]"
UNK = "[UNK]"

PAD_IND = 0
START_IND = 1
END_IND = 2
UNK_IND = 3


def all_num(token):
    n = "0123456789."
    for c in token:
        if c not in n:
            return False
    return True

def sents2batches(dataset, batch_size):
    batched_dataset = []
    sentence_lens = []
    current_len = 0
    i = 0

    ## they are already in sorted order
    current = []
    return [[x] for x in dataset], [[len(x.preprocessed.split(" "))] for x in dataset]


def group_into_batch(dataset, batch_size):
    """

        Batch size is given in word length so that some batches do not contain
        too many examples!!!

        Do not naively batch by number of sentences!!

    """

    batched_dataset = []
    sentence_lens = []
    current_len = 0
    i = 0

    ## they are already in sorted order
    current = []
    max_len = 0
    print("Batch size: {}".format(batch_size))
    for x in dataset:
        current.append(x)
        max_len = max(len(x), max_len)  ##
        current_len += len(x)
        if len(x) > 200:
            logging.info("Length {}".format(len(x)))
            logging.info(x)
        if current_len > batch_size:
            # print(current)
            current, lens = pad_trunc_nerdata_batch(current, max_len)
            batched_dataset.append(current)
            sentence_lens.append(lens)
            current = []
            current_len = 0
            max_len = 0
    if len(current) > 0:
        current, lens = pad_trunc_batch(current, max_len)
        sentence_lens.append(lens)
        batched_dataset.append(current)
    return batched_dataset, sentence_lens

class NerDataReader:

    def __init__(self, file_path, task_name,
                 tokenizer, batch_size=300,
                 for_eval=False, crf=False,
                 length_limit=10):
        self.for_eval = for_eval
        self.file_path = file_path
        self.iter = 0
        self.task_name = task_name
        self.batch_size = batch_size
        self.length_limit = length_limit
        self.dataset, self.orig_idx, self.label_counts = self.get_dataset()
        print("Dataset size : {}".format(len(self.dataset)))
        self.data_len = len(self.dataset)
        self.l2ind, self.word2ind, self.vocab_size = self.get_vocabs()
        # self.pos_voc = Vocab(self.pos2ind)
        self.label_vocab = Vocab(self.l2ind)
        self.word_vocab = Vocab(self.word2ind)
        self.batched_dataset, self.sentence_lens = sents2batches(self.dataset, batch_size=self.batch_size)
        self.for_eval = for_eval
        self.num_cats = len(self.l2ind)
        print("Number of NER categories: {}".format(self.num_cats))
        self.bert_tokenizer = tokenizer
        self.val_index = 0

    def get_ind2sent(self, sent):
        return " ".join([self.word2ind[w] for w in sent])

    def get_vocabs(self):
        l2ind = {PAD: PAD_IND, START_TAG: START_IND, END_TAG: END_IND}
        word2ix = {PAD: PAD_IND, START_TAG: START_IND, END_TAG: END_IND}
        for sent in self.dataset:
            for word in sent.words:
                if word not in word2ix:
                    word2ix[word] = len(word2ix)
            for l in sent.labels:
                if l not in l2ind:
                    l2ind[l] = len(l2ind)
        vocab_size = len(word2ix)
        return l2ind, word2ix, vocab_size

    def get_dataset(self):
        corpus = data_reader(self.file_path, encoding='utf-8', skip_unlabeled=self.skip_unlabeled)
        new_dataset = []
        for s in corpus:
            if len(s.words) < self.length_limit or len(s.words) > 200:
                continue
            # s.labels = [START_TAG] + s.labels + [END_TAG]
            # s.preprocessed = " ".join(START_TAG,s.preprocessed,END_TAG)
            new_dataset.append(s)
        return new_dataset, 0, 0

    ## compatible with getSent and for word embeddings
    def prepare_sent(self, sent, word2ix):
        idx = [word2ix[word[0]] for word in sent]
        return torch.tensor(idx, dtype=torch.long)

    def prepare_label(self, labs, l2ix):
        idx = [l2ix[lab] for lab in labs]
        return torch.tensor(idx, dtype=torch.long)

    def getword2vec(self, row):
        key = row[0].lower()
        root = row[1][:row[1].find("+")].encode().decode("unicode-escape")
        while (len(key) > 0):
            if key in word_vectors:
                return word_vectors[key]
            elif root.lower() in word_vectors:
                return word_vectors[root.lower()]
            else:
                return word_vectors["OOV"]
        return 0


    def getword2vec2(self, row):
        key = row[0].lower()
        root = row[1][:row[1].find("+")].encode().decode("unicode-escape")  ## for turkish special chars
        while (len(key) > 0):
            if key in word_vectors:
                return 2
            elif root.lower() in word_vectors:
                return 1
            else:
                return 0
        return 0

    def __len__(self):
        return len(self.batched_dataset)

    def __getitem__(self, idx, random=True):
        """
            Indexing for the DepDataset
            converts all the input into tensor before passing

            input is of form :
                word_ids  (Batch_size, Sentence_lengths)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if not self.for_eval:
            if random:
                idx = np.random.randint(len(self.batched_dataset))
            idx = idx % len(self.batched_dataset)
        batch = self.batched_dataset[idx]
        lens = self.sentence_lens[idx]
        tok_inds = []
        ner_inds = []
        tokens = []
        for x in batch:
            preprocessed = x.preprocessed
            labels = x.labels
            tokens.append(x.words)
            ner_inds.append(self.label_vocab.map(labels))
        i = 0
        max_bert_len = 0
        bert_input = self.bert_tokenizer([x.preprocessed for x in batch], truncation=True, padding=True,
                                         return_tensors="pt")
        return batch, bert_input


if __name__ == "__main__":
    # data_path = 'toy_ner_data.tsv'
    # reader = DataReader(data_path, "NER")
    # print(sum(map(len,reader.dataset))/reader.data_len)
    # batched_dataset, sentence_lens = group_into_batch(reader.dataset,batch_size = 300)
    ner_file_path = 'biobert_data/datasets/NER/BC2GM/ent_train.tsv'
    biobert_model_name = "dmis-lab/biobert-v1.1"
    length_limit = 10
    bert_tokenizer = BertTokenizer.from_pretrained(biobert_model_name)
    dataset = NerDataReader(ner_file_path, "NER", for_eval=True, tokenizer=bert_tokenizer,
                            batch_size=1, length_limit=length_limit)
    batch, bert_input = dataset[0]
    print(bert_input)
