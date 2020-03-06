import pandas as pd
from gensim.test.utils import datapath
from gensim import corpora, models, similarities
import json
import numpy as np
from lda_util import clean_tsv
import os

ROOT_DIR = os.path.abspath("")
MODEL_DIR = os.path.join(ROOT_DIR, "model")

#print(lda.print_topics(num_topics=20, num_words=20))

def load_json(json_dir):
    with open(json_dir, 'r') as f:
        return json.load(f)

def lda_sim(doc1,doc2, lda, topic_num=20):
    doc1_lda = lda[doc1]
    doc2_lda = lda[doc2]
    doc1_vector = np.zeros(topic_num)
    doc2_vector = np.zeros(topic_num)
    for topic in doc1_lda:
        doc1_vector[topic[0]] = topic[1]
    for topic in doc2_lda:
        doc2_vector[topic[0]] = topic[1]
    try:
        sim = np.dot(doc1_vector, doc2_vector) / \
            (np.linalg.norm(doc1_vector) * np.linalg.norm(doc2_vector))
    except ValueError:
        sim = 0
    return sim


if __name__ == '__main__':
    fname = datapath(os.path.join(MODEL_DIR, "lda_model"))
    lda = models.ldamodel.LdaModel.load(fname, mmap='r')
    texts_list = load_json('data_cleaned/hair_dryer_cleaned.json')

    dictionary = corpora.Dictionary(texts_list)
    corpus = [dictionary.doc2bow(texts) for texts in texts_list]
    sim = lda_sim(corpus[1],corpus[18],lda)
    print(sim)