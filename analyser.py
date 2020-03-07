import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

class Analyser:
    def __init__(self, path="data/hair_dryer.tsv"):
        # read raw data
        self.raw_df = pd.read_csv(path, sep='\t')

    def _get_product(self, product_parent):
        df = self.raw_df
        if product_parent != 0:
            try:
                df = df.groupby("product_parent").get_group(product_parent)
            except KeyError:
                raise ("Unexisted product_parent!")
        return df

    def average_rating(self):
        print(self.raw_df.groupby("product_parent").mean()["star_rating"])

    def rating_distribution(self, product_parent=0):
        '''
            ### Draw the picture of rating distribution of given product
            If product_parent = 0, it will draw the global distribution
        '''
        x_axis = [1, 2, 3, 4, 5]
        y_axis = [0, 0, 0, 0, 0]

        df = self._get_product(product_parent)

        star_rating = df["star_rating"].tolist()

        # in case of some rating hierarchy not exists
        # we traverse the list entirely
        for i in star_rating:
            y_axis[i-1] += 1

        plt.title("Product %d Rating Distribution" %product_parent)
        plt.xlabel("Rating")
        plt.ylabel("Number")
        plt.bar(x_axis, y_axis)
        plt.show()

    def voting_distribution(self, product_parent=0):
        '''
            ### Draw the picture of voting distribution of given product
            If product_parent = 0, it will draw the global distribution
        '''

        df = self._get_product(product_parent)

        total_voting = df["total_votes"].tolist()
        res = {}

        # in case of some rating hierarchy not exists
        # we traverse the list entirely
        for i in total_voting:
            if res.get(i) is None:
                res[i] = 1
            else:
                res[i] += 1


        plt.title("Product %d Total Voting Distribution" %product_parent)
        plt.xlabel("Total Voting Number")
        plt.ylabel("Number")
        plt.bar(res.keys(), res.values())
        plt.show()

    def helpful_voting_ratio_distribution(self, product_parent=0, min_total_votes=1):
        '''
            ### Draw the picture of helpful voting ratio distribution of given product
            If product_parent = 0, it will draw the global distribution.
            Minimum total votes can be provided for further restriction
        '''

        df = self._get_product(product_parent)

        total_voting = df["total_votes"].tolist()
        helpful_voting = df["helpful_votes"].tolist()

        data = []

        for i in range(len(total_voting)):
            if total_voting[i] < min_total_votes:
                continue
            else:
                data.append(helpful_voting[i] / total_voting[i])

        plt.title("Product %d Helpful Voting Ratio Distribution (Minimum Total Voting = %d)" %(product_parent, min_total_votes))
        plt.xlabel("Helpful Voting Ratio")
        plt.ylabel("Number")
        plt.hist(data, bins=40)
        plt.show()

    def score_distribution(self, path="scoreboard/hair_dryer_score.csv"):
        '''
            ### Draw the picture of score distribution of given score file
        '''

        df = pd.read_csv(path)
        scores = df["score"].tolist()

        plt.title("Score Distribution")
        plt.xlabel("Score")
        plt.ylabel("Number")
        plt.hist(scores, bins=10)
        plt.show()

    def generate_word_cloud(self, product_parent=0):
        '''
            ### Draw the word cloud of given product's review
            If product_parent = 0, it will draw the global word cloud
        '''
        df = self._get_product(product_parent)

        review = df["review_body"].tolist()
        text = ""
        for t in review:
            # delete br
            t = t.replace("<br />", "")
            text += t + " "

        w = WordCloud(width=1920, height=1080, scale=1.5, background_color="black")
        w.generate(text)
        plt.imshow(w)
        plt.axis("off")
        plt.show()
        
    

if __name__ == "__main__":
    ana = Analyser()
    ana.score_distribution()
