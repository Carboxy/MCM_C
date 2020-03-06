from stop_words import get_stop_words
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from gensim.test.utils import datapath
from gensim import corpora, models, similarities
import json
import numpy as np
#print(lda.print_topics(num_topics=20, num_words=20))


def clean_tsv(tsv_dir):
    print('Cleaning...')
    # load reviews
    df = pd.read_csv(tsv_dir, sep='\t', encoding="utf-8")
    reviews = df["review_body"].tolist()

    # tokenize
    tokens_list = []

    for review in reviews:
        tokens = word_tokenize(re.sub('[^\w ]', '', review))
        tokens_list.append(tokens)

    # delete stop words
    en_stop = get_stop_words('en')
    stopped_tokens_list = []

    for tokens in tokens_list:
        stopped_tokens = [token for token in tokens if not token in en_stop]
        stopped_tokens_list.append(stopped_tokens)

    # extract stem
    p_stemmer = PorterStemmer()
    texts_list = []

    for stopped_tokens in stopped_tokens_list:
        texts = [p_stemmer.stem(stopped_token) for stopped_token in stopped_tokens]
        texts_list.append(texts)
    return texts_list

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
    fname = datapath("D:/Academic/数学建模/mcm code/MCM_C-master/model/lda_model")
    lda = models.ldamodel.LdaModel.load(fname, mmap='r')
    texts_list = load_json('data_cleaned/hair_dryer_cleaned.json')

    dictionary = corpora.Dictionary(texts_list)
    corpus = [dictionary.doc2bow(texts) for texts in texts_list]
    sim = lda_sim(corpus[1],corpus[18],lda)
    print(sim)