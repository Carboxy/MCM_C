import pandas as pd
from lda_util import clean_tsv
from nltk.text import TextCollection
import math

class TF_IDF:
    def __init__(self, path="data/hair_dryer.tsv"):
        self.raw_df = pd.read_csv(path, sep='\t', encoding='utf-8')

    def get_tf_idf_dict(self, column_type="review_body", save_path="tf_idf_value/hair_dryer_tf_idf.csv"):
        '''
            Get TF-IDF dictionary through all the review
            You can choose review headline to analyse either
            (let column_type = "review_headline")
        '''
        reviews = self.raw_df[column_type].tolist()

        # get clean header
        reviews_list_cleaned = clean_tsv(reviews)

        freq = {}
        # traverse to get the word frequency
        for reviews in reviews_list_cleaned:
            for review in reviews:
                if freq.get(review) is None:
                    freq[review] = 1
                else:
                    freq[review] += 1

        # calculate tf
        total_freq = sum(freq.values())
        tf = {}

        for key in freq.keys():
            tf[key] = freq[key] / total_freq

        # calculate idf
        total_review = len(reviews_list_cleaned)
        doc_freq = {}
        idf = {}
        for word in freq.keys():
            doc_freq[word] = 0
            for reviews in reviews_list_cleaned:
                if word in reviews:
                    doc_freq[word] += 1

        for word in doc_freq.keys():
            idf[word] = math.log(total_review / (doc_freq[word] + 1))

        # calculate tf-idf
        words = []
        tf_idf = []
        for word in tf.keys():
            words.append(word)
            tf_idf.append(tf[word] * idf[word])

        df = pd.DataFrame({"word": words, "tf-idf": tf_idf})
        df.to_csv(save_path, encoding='utf-8')


    def get_tf_idf_dict_nltk(self, column_type="review_body", save_path="tf_idf_value/hair_dryer_tf_idf.csv"):
        '''
            ### nltk version
            it's super slow so don't use it
        '''
        reviews = self.raw_df[column_type].tolist()

        # get clean header
        reviews_list_cleaned = []
        for review in reviews:
            reviews_list_cleaned.append(clean_review(review))

        # get all words
        words = set()
        for reviews in reviews_list_cleaned:
            for review in reviews:
                words.add(review)

        words = list(words)

        corpus = TextCollection(reviews_list_cleaned)

        tf_idf = []
        for word in words:
            tf_idf.append(corpus.tf_idf(word, corpus))

        df = pd.DataFrame({"word": words, "tf-idf": tf_idf})
        df.to_csv(save_path, encoding='utf-8')
        

if __name__ == "__main__":
    test = TF_IDF(path="data/pacifier_filtered.tsv")
    test.get_tf_idf_dict(save_path="tf_idf_value/pacifier_filtered_tf_idf.csv")

