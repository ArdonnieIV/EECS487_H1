# EECS 487 Intro to NLP
# Assignment 1

import math
import random
from collections import defaultdict
from itertools import product
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


class NGramLM:
    """N-gram language model."""

    def __init__(self, bos_token, eos_token, tokenizer, ngram_size):
        self.ngram_count = {}
        for i in range(ngram_size):
            # Each of these will be its own dictionary containing all of the (i-1)-grams
            # For instance, self.ngram_count[0] will be its own dictionary containing all unigrams
            self.ngram_count[i] = None

        self.ngram_size = ngram_size

        self.vocab_sum = None # could be useful in linear interpolation
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.tokenizer = tokenizer

    def get_ngram_counts(self, reviews):

        # Set up
        for i in range(self.ngram_size):
            self.ngram_count[i] = defaultdict(int)

        # Count
        for review in reviews:
  
            tokens = self.tokenizer(review.lower())
            tokens.append(self.eos_token)
            for i in range(2 + (self.tokenizer != word_tokenize)):
                tokens.insert(0, self.bos_token)

            for i in range(self.ngram_size):
                for j in range(len(tokens) - i):
                    thisKey = tokens[j] if i == 0 else tuple(tokens[j:j+i+1])
                    self.ngram_count[i][thisKey] += 1

        # Clean up
        toRemove = []
        self.vocab_sum = 0
        self.ngram_count[0]['UNK'] = 0

        for token in self.ngram_count[0]:

            if token != '<s>':
                self.vocab_sum += self.ngram_count[0][token] # Should you count '<s>'?

            if self.ngram_count[0][token] < 2:
                toRemove.append(token)
                self.ngram_count[0]['UNK'] += self.ngram_count[0][token]
        
        for token in toRemove:
            self.ngram_count[0].pop(token)

            # TODO - UNK tokens should trickle up
            for i in range(1, self.ngram_size):
                for bigToken in self.ngram_count[i]:
                    if token in bigToken:
                        print(token, bigToken)

        #################################################################################

    def add_k_smooth_prob(self, n_minus1_gram, unigram, k):

        V = len(self.ngram_count[0]) - 1
        gramIndex = len(n_minus1_gram) - 1 if type(n_minus1_gram) == tuple else 0

        probability = self.ngram_count[gramIndex + 1][n_minus1_gram + (unigram,)] + k
        probability /= (self.ngram_count[gramIndex][n_minus1_gram] + k*V)

        return probability
        
        #################################################################################

    def linear_interp_prob(self, n_minus1_gram, unigram, lambdas):

        probability = 0
        gramIndex = len(n_minus1_gram) - 1 if type(n_minus1_gram) == tuple else 0

        for i, lam in enumerate(lambdas):

            thisProb = n_minus1_gram[i:] + (unigram,) if len(n_minus1_gram[i:]) > 0 else unigram

            if type(thisProb) == tuple:
                denom = thisProb[:-1] if len(thisProb[:-1]) > 1 else thisProb[:-1][0]
                print(thisProb, denom)
                print(self.ngram_count[gramIndex+1-i][thisProb], self.ngram_count[gramIndex-i][denom])
                probability += lam * (self.ngram_count[gramIndex+1-i][thisProb] / self.ngram_count[gramIndex-i][denom])
            else:
                probability += lam * (self.ngram_count[0][thisProb] / self.vocab_sum)

        return probability

        #################################################################################
    
    def get_probability(self, n_minus1_gram, unigram, smoothing_args):

        probability = 0

        if smoothing_args['method'] == 'add_k':
            probability = self.add_k_smooth_prob(n_minus1_gram, unigram, smoothing_args['k'])
        else:
            probability = self.linear_interp_prob(n_minus1_gram, unigram, smoothing_args['lambdas'])

        return probability

        #################################################################################
    
    def get_perplexity(self, text, smoothing_args):
        perplexity = 0

        #################################################################################
        # TODO: calculate perplexity for text
        #################################################################################

        #################################################################################

        return perplexity
    
    def search_k(self, dev_data):
        best_k = 0

        #################################################################################
        # TODO: find best k value
        #################################################################################

        #################################################################################

        return best_k
    
    def search_lambda(self, dev_data):
        best_lambda = [0, 0, 0]

        #################################################################################
        # TODO: find best lambda values
        #################################################################################

        #################################################################################

        return best_lambda
    
    def generate_text(self, prompt, smoothing_args):
        generated_text = prompt.copy()

        #################################################################################
        # TODO: generate text based on prompt
        #################################################################################

        #################################################################################

        print(' '.join(generated_text))

def load_new_data(df):

    df_class1 = None
    df_class2 = None

    #################################################################################
    # TODO: load the reviews based on a split of your choosing
    #################################################################################

    #################################################################################

    display(df_class1[["text", split_name]])
    display(df_class2[["text", split_name]])

    return (class1_trn, class1_dev, class2_trn, class2_dev)
    

def predict_class(test_file, class1_lm, class2_lm, smoothing_args):

    class1_ppl = 0
    class2_ppl = 0

    #################################################################################
    # TODO: load the review in test_file, predict its class
    #################################################################################

    #################################################################################

    print(f"Perplexity for class1_lm: {class1_ppl}")
    print(f"Perplexity for class2_lm: {class2_ppl}")
    if class1_ppl < class2_ppl:
        print("It is in class 1")
    else:
        print("It is in class 2")
