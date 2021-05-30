from gensim.models import FastText , KeyedVectors
p = "../dataselection_data/bio_embedding_extrinsic"


from gensim.models.wrappers import fasttext as ft_wrapper


wv = KeyedVectors.load_word2vec_format(p,binary=True)
