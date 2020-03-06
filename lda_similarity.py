from stop_words import get_stop_words
import pandas as pd
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
import re
from gensim.test.utils import datapath
from gensim import corpora, models, similarities
import json
import numpy as np
import sys

def load_json(json_dir):
    with open(json_dir, 'r') as f:
        return json.load(f)

def get_corpus():

    # load reviews
    df = pd.read_csv("data/hair_dryer.tsv", sep='\t', encoding="utf-8")
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

    dictionary = corpora.Dictionary(texts_list)
    dictionary.save('model/lda/lda.dict')
    corpus = [dictionary.doc2bow(texts) for texts in texts_list]
    return corpus

class lda_sim:
    def __init__(self, model_dir, index_dir, dict_dir, texts_list_dir, topic_num=20):
        self.model = models.ldamodel.LdaModel.load(model_dir)
        self.index = similarities.Similarity.load(index_dir)
        self.dictionary = corpora.Dictionary.load(dict_dir)
        self.texts_list = load_json(texts_list_dir)
        self.corpus = [self.dictionary.doc2bow(texts) for texts in self.texts_list]
        #self.corpus = get_corpus()

    def get_lda_sim(self, ref_idxs, topN=20):
        '''
        对于一系列给定的参考评价，搜索所有评价中与参考评价相似度最高的topN条，返回
        索引和相似度
        Args:
            ref_dir: 参考评价的路径
        '''
        sim_sum = np.zeros(len(self.corpus))
        count = 0
        for ref_idx in ref_idxs:
            ref_bow = self.corpus[ref_idx]
            try:
                sim_sum += self.index[self.model[ref_bow]]
                count += 1
            except:
                print('数组越界')
                continue
        ind = np.argpartition(sim_sum, -topN)[-topN:]
        return sim_sum[ind]/count, ind


if __name__ == '__main__':
    model_dir = 'model/lda/lda1.model'
    index_dir = 'model/lda/lda1.index'
    dict_dir = 'model/lda/lda1.dict'
    texts_list_dir = 'data_cleaned/hair_dryer_cleaned.json'
    ref_dir = 'reference_review/good.json'
    good_idx = [11383, 11384, 11398, 11429, 11442, 11448, 11463, 11468, 11469]
    bad_idx = [0, 20, 1252, 1296, 4796, 5618, 6134, 6844, 8084, 8130]
    lda_sim = lda_sim(model_dir, index_dir, dict_dir, texts_list_dir)
    score, ind = lda_sim.get_lda_sim(bad_idx, 20)
    print(score)
    print(ind)  
