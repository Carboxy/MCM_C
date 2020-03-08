import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lda_util import clean_review, clean_tsv

# REVISE ALL PARAMETERS HERE!

# $m = 0, value = HVR_BASE_SCORE$
# $0 < m < HVR_MIN_PEOPLE, value = 50 + (n - 0.5m) \times HVR_PARA_1$
# $m \geq 5, value = 100 * \dfrac{n}{m}$
HVR_BASE_SCORE = 40
HVR_MIN_PEOPLE = 5
HVR_PARA_1 = 20

# score = VINE_BASE + value_hvr * VINE_PARA
VINE_BASE = 80
VINE_PARA = 0.2

# # $l \in N^+$
# # $l \leq BODY_THRES_1, value = BODY_PARA_1 \times l$
# # $BODY_THRES_1 < l < BODY_THRES_2, value = BODY_PARA_2 \times l - BODY_PARA_3$
# # $l \geq BODY_THRES_2, value = 100$
# BODY_THRES_1 = 10
# BODY_PARA_1 = 0.5
# BODY_THRES_2 = 20
# BODY_PARA_2 = 10
# BODY_PARA_3 = 100

DEFAULT_HEADER_PARA = 0.1
VERIFIED_PURCHASE_PARA = 0.5


class Score:
    def __init__(self, path="ml_dataset/hair_dryer_predict.csv"):
        # read raw data
        self.raw_df = pd.read_csv(path)

    def _cal_helpful_review_ratio_value(self, row):
        '''
            turn helpful_review_ratio into value [0, 100]
        '''
        m = row['total_votes']
        n = row['helpful_votes']

        if m == 0:
            return HVR_BASE_SCORE
        elif m < HVR_MIN_PEOPLE:
            return 50 + (n - 0.5 * m) * HVR_PARA_1
        else:
            return 100 * n / m

    def _cal_review_body_value(self, row):
        '''
            turn length of review_body into value [0, 100]
        '''
        # text = row["review_body"]
        # word_list = clean_review(text)
        # l = len(word_list)
        # if l <= BODY_THRES_1:
        #     return BODY_PARA_1 * l
        # elif l < BODY_THRES_2:
        #     return BODY_PARA_2 * l - BODY_PARA_3
        # else:
        #     return 100
        return row["predict"]

    def _is_default_header(self, row):
        header = row["review_headline"]
        return header == "Five Stars" or \
               header == "Four Stars" or \
               header == "Three Stars" or \
               header == "Two Stars" or \
               header == "One Star"

    def calc_score(self):
        # traverse through row
        scores = []
        for index, row in self.raw_df.iterrows():
            score = 0
            if row['vine'] == 'Y':
                score = VINE_BASE + self._cal_helpful_review_ratio_value(row) * VINE_PARA
            else:
                para = 1
                if self._is_default_header(row):
                    para *= DEFAULT_HEADER_PARA
                if row["verified_purchase"] == "N":
                    para *= VERIFIED_PURCHASE_PARA
                score = self._cal_helpful_review_ratio_value(row) * para * self._cal_review_body_value(row)
            scores.append(score)

        self.raw_df["score"] = scores

    def save(self, path="scoreboard/hair_dryer_score.csv"):
        self.raw_df.to_csv(path)

    def draw_tf_idf_distribution(self, path="tf_idf_value/hair_dryer_tf_idf_dict.csv", save_path="tf_idf_value/hair_dryer_tf_idf.csv"):
        reviews = self.raw_df["review_body"].tolist()
        reviews_list_cleaned = clean_tsv(reviews)

        df = pd.read_csv(path)
        # tf-idf: list of dict
        tf_idf = df.to_dict(orient="records")

        data = []

        for reviews in reviews_list_cleaned:
            total = 0
            count = 0
            result = 0
            for review in reviews:
                for item in tf_idf:
                    if review == item["word"]:
                        total += item["tf-idf"]
                        count += 1
                        break
            if count != 0:
                result = total / count
            data.append(result)

        self.raw_df["tf-idf"] = data
        self.raw_df.to_csv(save_path)
        
        plt.title("TF-IDF Distribution")
        plt.xlabel("AVG TF-IDF")
        plt.ylabel("Number")
        plt.hist(data, bins=100)
        plt.show()


if __name__ == "__main__":
    path = "ml_dataset/pacifier_filtered_predict.csv"
    save_path = "scoreboard/pacifier_filtered_score.csv"
    s = Score(path)
    s.calc_score()
    s.save(save_path)

   

    