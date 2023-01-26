# EECS 487 Intro to NLP
# Assignment 1

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def load_headlines(filename):

    df = pd.read_csv(filename)
    return df

    ###################################################################


def get_basic_stats(df):

    avg_len = 0
    std_len = 0
    num_articles = {0: 0, 1: 0}

    num_articles[1] = df['is_clickbait'].sum()
    num_articles[0] = df.shape[0] - num_articles[1]

    numTokens = []
    for ind in df.index:
        numTokens.append(len(df['text'][ind].split(' ')))

    avg_len = sum(numTokens) / len(numTokens)
    variance = sum([((x - avg_len) ** 2) for x in numTokens]) / len(numTokens)
    std_len = variance ** 0.5

    print(f"Average number of tokens per headline: {avg_len}")
    print(f"Standard deviation: {std_len}")
    print(f"Number of legitimate/clickbait headlines: {num_articles}")

    ###################################################################


class NaiveBayes:
    """Naive Bayes classifier."""

    def __init__(self):
        self.ngram_count = []
        self.total_count = []
        self.category_prob = []
    
    def fit(self, data):

        self.vectorizer = CountVectorizer(max_df=0.8, min_df=3, ngram_range=(1,2))
        self.vectorizer.fit(data['text'].str.lower())

        for i in range(2):
            self.ngram_count.append(self.vectorizer.transform(data.loc[data['is_clickbait'] == i]['text'].str.lower()).sum(axis=0))
            self.total_count.append(self.ngram_count[i].sum())
            self.category_prob.append(len(data.loc[data['is_clickbait'] == i]) / len(data))

        ###################################################################
    
    def calculate_prob(self, docs, c_i):

        prob = []

        vectors = self.vectorizer.transform(doc.lower() for doc in docs)

        for vector in vectors:

            thisProb = 0

            for i, value in enumerate(vector.toarray()[0]):
                if value:
                    thisProb += np.log((self.ngram_count[c_i][0, i] + 1) / (self.total_count[c_i] + len(self.vectorizer.vocabulary_)))*value
            
            thisProb += np.log(self.category_prob[c_i])
            prob.append(np.exp(thisProb))

        return prob

        ###################################################################


    def predict(self, docs):

        prediction = [None] * len(docs)
        allProbs = []

        for i in range(2):
            allProbs.append(self.calculate_prob(docs, i))
        
        for i in range(len(docs)):
            if allProbs[0][i] > allProbs[1][i]:
                prediction[i] = 0
            else:
                prediction[i] = 1

        return prediction

        ###################################################################


def evaluate(predictions, labels):

    accuracy, mac_f1, mic_f1 = None, None, None
    cM = np.array([[0, 0], [0, 0]])
    
    for i in range(len(predictions)):
        cM[predictions[i], labels[i]] += 1

    accuracy = np.trace(cM) / cM.sum()
    precision = cM[1, 1] / (cM[1, 1] + cM[1, 0])
    recal = cM[1, 1] / (cM[1, 1] + cM[0, 1])
    mac_f1 = mic_f1 = (2*precision*recal) / (precision + recal)

    return accuracy, mac_f1, mic_f1

    ###################################################################

