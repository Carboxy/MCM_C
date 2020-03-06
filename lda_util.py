# blog source
# https://blog.csdn.net/github_36299736/article/details/54966460

import pandas as pd
import nltk
import re
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
from gensim.test.utils import datapath


# load reviews
df = pd.read_csv("data/hair_dryer.tsv", sep='\t', encoding="utf-8")
reviews = df["review_body"].tolist()

# tokenize
tokens_list = []

for review in reviews:
    review = review.replace("<br />", " ")
    tokens = nltk.word_tokenize(re.sub('[^\w ]', '', review))
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

dictionary = corpora.Dictionary(texts_list)
corpus = [dictionary.doc2bow(texts) for texts in texts_list]
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20) 
temp_file = datapath("model/lda_model")
ldamodel.save(temp_file)

