# blog source
# https://blog.csdn.net/zaishijizhidian/article/details/89365733

from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.svm import SVC
import sys
from lda_util import clean_review

class SVM_Model:
    def __init__(self, product="hair_dryer"):
        path_high_qlt = "ml_dataset/" + product + "_high_quality.csv"
        path_low_qlt = "ml_dataset/" + product + "_low_quality.csv"
        self.high_qlt_df = pd.read_csv(path_high_qlt)
        self.low_qlt_df = pd.read_csv(path_low_qlt)
        self.product = product
        self.n_dim = 100

    def get_training_set(self):
        '''
            get training set and testing set
        '''
        high_qlt_reviews = self.high_qlt_df["review_body"].apply(clean_review)
        low_qlt_reviews = self.low_qlt_df["review_body"].apply(clean_review)

        # get tag   
        # 1: high quality, 0: low quality
        y = np.concatenate((np.ones(len(high_qlt_reviews)), np.zeros(len(low_qlt_reviews))))
        x_train, x_test, y_train, y_test = train_test_split(np.concatenate((high_qlt_reviews, low_qlt_reviews)), y, test_size=0.2)
        # print(x_train.shape)
        # print(x_test.shape)
        # print(y_train.shape)
        # print(y_test.shape)
        np.save("ml_dataset/" + self.product + "_y_train.npy", y_train)
        np.save("ml_dataset/" + self.product + "_y_test.npy", y_test)
        return x_train, x_test

    def build_sentence_vector(self, text, size, imdb_w2v):
        vec = np.zeros(size).reshape((1, size))
        count = 0.
        for word in text:
            try:
                vec += imdb_w2v[word].reshape((1, size))
                count += 1.
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec

    def get_train_vecs(self, x_train, x_test):
        # initialize model and vocabulary
        imdb_w2v = Word2Vec(size=self.n_dim, min_count=50)
        imdb_w2v.build_vocab(x_train)
        
        # training
        imdb_w2v.train(x_train, total_examples=imdb_w2v.corpus_count, epochs=500)
        
        train_vecs = np.concatenate([self.build_sentence_vector(text, self.n_dim, imdb_w2v) for text in x_train])
        # train_vecs = scale(train_vecs)
        
        np.save("ml_dataset/" + self.product + "_train_vecs.npy", train_vecs)
        # print(train_vecs.shape)

        # training on testing set
        imdb_w2v.train(x_test, total_examples=imdb_w2v.corpus_count, epochs=imdb_w2v.epochs)
        imdb_w2v.save("ml_dataset/" + self.product + "_w2v_model.pkl")

        # Build test tweet vectors then scale
        test_vecs = np.concatenate([self.build_sentence_vector(text, self.n_dim, imdb_w2v) for text in x_test])
        # test_vecs = scale(test_vecs)
        np.save("ml_dataset/" + self.product + "_test_vecs.npy", test_vecs)
        # print(test_vecs.shape)

    def get_data(self):
        train_vecs = np.load("ml_dataset/" + self.product + "_train_vecs.npy")
        y_train = np.load("ml_dataset/" + self.product + "_y_train.npy")
        test_vecs = np.load("ml_dataset/" + self.product + "_test_vecs.npy")
        y_test = np.load("ml_dataset/" + self.product + "_y_test.npy")
        return train_vecs, y_train, test_vecs, y_test

    def svm_train(self, train_vecs, y_train, test_vecs, y_test):
        clf = SVC(kernel='rbf', verbose=True, probability=True)
        clf.fit(train_vecs, y_train)
        joblib.dump(clf, "ml_dataset/" + self.product + "_svm_model.pkl")
        # print accuracy on test dataset
        print(clf.score(test_vecs, y_test))

    def get_predict_vecs(self, words):
        imdb_w2v = Word2Vec.load("ml_dataset/" + self.product + "_w2v_model.pkl")
        # imdb_w2v.train(words)
        train_vecs = self.build_sentence_vector(words, self.n_dim, imdb_w2v)
        # print train_vecs.shape
        return train_vecs

    def svm_predict(self, string):
        words = clean_review(string)
        words_vecs = self.get_predict_vecs(words)
        clf = joblib.load("ml_dataset/" + self.product + "_svm_model.pkl")
        
        result = clf.predict_proba(words_vecs)
        return result

    def predict_tsv(self):
        path = "data/" + self.product + ".tsv"
        df = pd.read_csv(path, sep='\t')

        results = []
        clf = joblib.load("ml_dataset/" + self.product + "_svm_model.pkl")
        imdb_w2v = Word2Vec.load("ml_dataset/" + self.product + "_w2v_model.pkl")

        for index, row in df.iterrows():
            words = clean_review(row["review_body"])
            words_vecs = self.build_sentence_vector(words, self.n_dim, imdb_w2v)
            result = clf.predict_proba(words_vecs)[0][1]

            results.append(result)

        save_path = "ml_dataset/" + self.product + "_predict.csv"
        df["predict"] = results
        df.to_csv(save_path)


if __name__ == "__main__":
    m = SVM_Model(product="pacifier_filtered")
    # x_train, x_test = m.get_training_set()
    # m.get_train_vecs(x_train, x_test)

    # train_vecs, y_train, test_vecs, y_test = m.get_data()
    # m.svm_train(train_vecs, y_train, test_vecs, y_test)

    m.predict_tsv()

