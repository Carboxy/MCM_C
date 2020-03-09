import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

class Reputation:
    def __init__(self, path="scoreboard/hair_dryer_score.csv"):
        self.raw_df = pd.read_csv(path)

    def _star2score(self, star):
        '''
            turn number of stars into score
            1: 20, 2: 40, 3: 60, 4: 80, 5: 100
        '''
        return 20 * star

    def _str2date(self, s):
        s = s.split("/")
        year = int(s[2])
        month = int(s[0])
        day = int(s[1])
        return datetime.datetime(year, month, day)

    def draw_total_reputation_tendency(self, product_parent):
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
            total_reputation += (df.star_rating[idx] + float(df.review_score[idx])) / 2 * score
            x_axis.append(df.review_date[idx])
            y_axis.append(total_reputation / total_score)

        plt.figure(dpi=80)
        plt.title("Product %d Reputation Tendency" %(product_parent))
        plt.xlabel("Time")
        plt.ylabel("Reputation")
        a = list(range(len(x_axis)))
        plt.xticks(a, rotation=90)
        plt.plot(x_axis, y_axis, color="b")
        

        plt.show()

    def draw_reputation_tendency_flex_version(self, product_parent, step):
        df = self.raw_df.groupby("product_parent").get_group(product_parent)

        scores = []
        single_reputation = []
        time = []

        for idx in reversed(df.index):
            score = float(df.score[idx])
            scores.append(score)
            single_reputation.append((df.star_rating[idx] + float(df.review_score[idx])) / 2 * score)
            time.append(df.review_date[idx])

        for _ in range(int(step / 2)):
            time.pop(-1)

        for _ in range(int((step - 1) / 2)):
            time.pop(0)

        y_axis = []

        for idx in range(len(single_reputation) - (step - 1)):
            y_axis.append(sum(single_reputation[idx:idx+step]) / sum(scores[idx:idx+step]))
        

        plt.figure(dpi=80)
        plt.title("Product %d Reputation Tendency (Flexible Version) (step = %d)" %(product_parent, step))
        plt.xlabel("Time")
        plt.ylabel("Reputation")
        a = list(range(len(time)))
        plt.xticks(a, rotation=90)
        plt.plot(time, y_axis, color="green")

        plt.show()

    def draw_reputation_tendency_flex_version_star_only(self, product_parent, step):
        df = self.raw_df.groupby("product_parent").get_group(product_parent)

        scores = []
        single_reputation = []
        time = []

        for idx in reversed(df.index):
            score = float(df.score[idx])
            scores.append(score)
            single_reputation.append(df.star_rating[idx] * score)
            time.append(df.review_date[idx])

        for _ in range(int(step / 2)):
            time.pop(-1)

        for _ in range(int((step - 1) / 2)):
            time.pop(0)

        y_axis = []

        for idx in range(len(single_reputation) - (step - 1)):
            y_axis.append(sum(single_reputation[idx:idx+step]) / sum(scores[idx:idx+step]))
            

        sns.set_style("whitegrid")
        plt.figure(dpi=80)
        # plt.title("Hair Dryer Product %d Reputation Tendency (step = %d)" %(product_parent, step))
        plt.xlabel("Time")
        plt.ylabel("Reputation")
        a = list(range(len(time)))
        plt.xticks(a, rotation=90)
        plt.plot(time, y_axis, color="b")

        plt.show()
        

if __name__ == "__main__":
    r = Reputation(path="scoreboard/microwave_score.csv")
    r.draw_reputation_tendency_flex_version_star_only(692404913, 23)
