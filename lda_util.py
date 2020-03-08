# blog source
# https://blog.csdn.net/github_36299736/article/details/54966460

import pandas as pd
import nltk
import re
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from gensim.test.utils import datapath
import os

ROOT_DIR = os.path.abspath("")
MODEL_DIR = os.path.join(ROOT_DIR, "model")

def clean_review(review):
    review = str(review).replace("<br />", " ")
    review = review.lower()
    tokens = nltk.word_tokenize(re.sub('[^\w ]', '', review))
    return tokens


def clean_tsv(reviews):
    print('Cleaning...')
    
    # tokenize
    tokens_list = []

    for review in reviews:
        tokens = clean_review(review)
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

    print('End Cleaning...')

    return texts_list

if __name__ == "__main__":
    df = pd.read_csv("data/hair_dryer.tsv", sep='\t')
    texts_list = clean_tsv(df["review_body"].tolist())

    dictionary = corpora.Dictionary(texts_list)
    corpus = [dictionary.doc2bow(texts) for texts in texts_list]
    ldamodel = models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20) 
    temp_file = datapath(os.path.join(MODEL_DIR, "lda_model"))
    ldamodel.save(temp_file)

