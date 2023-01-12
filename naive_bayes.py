# EECS 487 Intro to NLP
# Assignment 1

import pandas as pd
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def load_headlines(filename):
    df = None

    ###################################################################
    # TODO: load data into pandas dataframe
    ###################################################################

    ###################################################################

    return df


def get_basic_stats(df):
    avg_len = 0
    std_len = 0
    num_articles = {0: 0, 1: 0}

    ###################################################################
    # TODO: calculate mean and std of the number of tokens in the data
    ###################################################################

    ###################################################################

    print(f"Average number of tokens per headline: {avg_len}")
    print(f"Standard deviation: {std_len}")
    print(f"Number of legitimate/clickbait headlines: {num_articles}")


class NaiveBayes:
    """Naive Bayes classifier."""

    def __init__(self):
        self.ngram_count = []
        self.total_count = []
        self.category_prob = []
    
    def fit(self, data):

        ###################################################################
        # TODO: store ngram counts for each category in self.ngram_count
        ###################################################################

        pass

        ###################################################################
    
    def calculate_prob(self, docs, c_i):
        prob = None

        ###################################################################
        # TODO: calculate probability of category c_i given each headline in docs
        ###################################################################

        ###################################################################

        return prob

    def predict(self, docs):
        prediction = [None] * len(docs)

        ###################################################################
        # TODO: predict categories for the headlines
        ###################################################################

        ###################################################################

        return prediction


def evaluate(predictions, labels):
    accuracy, mac_f1, mic_f1 = None, None, None

    ###################################################################
    # TODO: calculate accuracy, macro f1, micro f1
    # Note: you can assume labels contain all values from 0 to C - 1, where
    # C is the number of categories
    ###################################################################

    ###################################################################

    return accuracy, mac_f1, mic_f1
