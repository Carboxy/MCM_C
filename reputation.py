import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Reputation:
    def __init__(self, path="scoreboard/hair_dryer_score.csv"):
        self.raw_df = pd.read_csv(path)

    def _star2score(self, star):
        '''
            turn number of stars into score
            1: 20, 2: 40, 3: 60, 4: 80, 5: 100
        '''
        return 20 * star

    def draw(self, product_parent):
        '''
            By given product, analyse the relation between time and
            total reputation, and draw the graph in the end
        '''
        df = self.raw_df.groupby("product_parent").get_group(product_parent)

        # since raw data is sorted by time, we don't need to sort again,
        # but we need to traverse reversedly

        total_reputation = 0
        total_score = 0
        x_axis = []
        y_axis = []
        for idx in reversed(df.index):
            score = float(df.score[idx])
            total_score += score
            total_reputation += (self._star2score(df.star_rating[idx]) + float(df.review_score[idx])) / 2 * score
            x_axis.append(df.review_date[idx])
            y_axis.append(total_reputation / total_score)

        plt.figure(dpi=80)
        plt.title("Product %d Reputation Tendency" %(product_parent))
        plt.xlabel("Time")
        plt.ylabel("Reputation")
        a = list(range(len(x_axis)))
        plt.xticks(a, rotation=90)
        plt.plot(x_axis, y_axis)
        plt.show()
        

if __name__ == "__main__":
    r = Reputation()
    r.draw(593915883)
