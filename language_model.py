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

        # Count unigrams
        for review in reviews:
            
            self.ngram_count[0][self.eos_token] += 1
            self.ngram_count[0][self.bos_token] += (2 + (self.tokenizer != word_tokenize))

            tokens = self.tokenizer(review.lower())
            for token in tokens:
                self.ngram_count[0][token] += 1

        # Replace infrequent unigrams with UNK
        toRemove = []
        self.vocab_sum = 0
        self.ngram_count[0]['UNK'] = 0
        for token in self.ngram_count[0]:
            if token != 'UNK':

                self.vocab_sum += self.ngram_count[0][token]

                if self.ngram_count[0][token] < 2:
                    toRemove.append(token)
                    self.ngram_count[0]['UNK'] += self.ngram_count[0][token]

        for word in toRemove:
            self.ngram_count[0].pop(word)

        # Count remaining n-grams
        for review in reviews:
  
            tokens = self.tokenizer(review.lower())
            for i, token in enumerate(tokens):
                if token in toRemove:
                    tokens[i] = 'UNK'

            tokens.append(self.eos_token)
            for i in range(2 + (self.tokenizer != word_tokenize)):
                tokens.insert(0, self.bos_token)

            for i in range(1, self.ngram_size):
                for j in range(len(tokens) - i):
                    self.ngram_count[i][tuple(tokens[j:j+i+1])] += 1

        #################################################################################

    def replace_unk(self, n_minus1_gram, unigram):

        # Check unigram
        if self.ngram_count[0][unigram] == 0:
            unigram = 'UNK'

        # Check n_minus1_gram
        n_minus1_fix = None
        if type(n_minus1_gram) == tuple:
            n_minus1_fix = list(n_minus1_gram)
            for i, token in enumerate(n_minus1_fix):
                if self.ngram_count[0][token] == 0:
                    n_minus1_fix[i] = 'UNK'
            n_minus1_fix = tuple(n_minus1_fix)
        else:
            if self.ngram_count[0][n_minus1_gram] == 0:
                n_minus1_fix = tuple('UNK')
            else:
                n_minus1_fix = tuple(n_minus1_gram)

        return n_minus1_fix, unigram

        #################################################################################

    def add_k_smooth_prob(self, n_minus1_gram, unigram, k): # TODO - What exactly happens to unknown words here?

        # Replace unknown tokens
        n_minus1_gram, unigram = self.replace_unk(n_minus1_gram, unigram)

        # Set parameters such as V
        V = len(self.ngram_count[0]) - 1
        gramIndex = len(n_minus1_gram) - 1 if type(n_minus1_gram) == tuple else 0

        # Execute formula
        probability = self.ngram_count[gramIndex + 1][n_minus1_gram + (unigram,)] + k
        probability /= (self.ngram_count[gramIndex][n_minus1_gram] + k*V)

        return probability
        
        #################################################################################

    def linear_interp_prob(self, n_minus1_gram, unigram, lambdas):

        # Replace unknown tokens
        n_minus1_gram, unigram = self.replace_unk(n_minus1_gram, unigram)

        probability = 0
        gramIndex = len(n_minus1_gram) - 1 if type(n_minus1_gram) == tuple else 0

        for i, lam in enumerate(lambdas):

            thisProb = n_minus1_gram[i:] + (unigram,) if len(n_minus1_gram[i:]) > 0 else unigram

            if type(thisProb) == tuple:
                denom = thisProb[:-1] if len(thisProb[:-1]) > 1 else thisProb[:-1][0]
                numerator = self.ngram_count[gramIndex+1-i][thisProb]
                if numerator:
                    probability += lam * (numerator / self.ngram_count[gramIndex-i][denom])
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

        tokens = self.tokenizer(text.lower())
        for i, token in enumerate(tokens):
            if self.ngram_count[0][token] == 0:
                tokens[i] = 'UNK' 
            
        tokens.append(self.eos_token)
        N = len(tokens)
        for i in range(2 + (self.tokenizer != word_tokenize)):
            tokens.insert(0, self.bos_token)

        for i in range(len(tokens) - self.ngram_size):
            n_minus1_gram = tuple(tokens[i:i+self.ngram_size-1])
            unigram = tokens[i+self.ngram_size-1]
            prob = self.get_probability(n_minus1_gram, unigram, smoothing_args)
            if prob:
                perplexity += math.log(prob)

        perplexity /= -N

        return math.exp(perplexity)

        #################################################################################
    
    def search_k(self, dev_data):

        best_k = 0
        best_p = float('inf')

        for this_k in [0.2, 0.4, 0.6, 0.8, 1.0]:

            this_p = 0
            for text in dev_data:
                this_p += self.get_perplexity(text, {'method': 'add_k','k': this_k})
            this_p /= len(dev_data)

            print(this_k, this_p)

            if this_p < best_p:
                best_p = this_p
                best_k = this_k

        return best_k

        #################################################################################

    
    def search_lambda(self, dev_data):

        search_space = [0.2, 0.4, 0.6, 0.8, 1.0]
        best_lambda = [0, 0, 0]
        best_p = float('inf')
        allLambdas = []

        for first in search_space:
            for second in search_space:
                for third in search_space:
                    allLambdas.append([first, second, third])

        for this_lambda in allLambdas:

            this_p = 0
            for text in dev_data:
                this_p += self.get_perplexity(text, {'method': 'linear','lambdas': this_lambda})
            this_p /= len(dev_data)

            if this_p < best_p:
                best_p = this_p
                best_lambda = this_lambda

        print(best_lambda, best_p)

        return best_lambda

        #################################################################################

    
    def generate_text(self, prompt, smoothing_args):

        tokens = prompt.copy()
        for i, token in enumerate(tokens):
            if self.ngram_count[0][token.lower()] == 0:
                tokens[i] = 'UNK' 
            else:
                tokens[i] = token.lower()

        for i in range(2 + (self.tokenizer != word_tokenize)):
            tokens.insert(0, self.bos_token)

        while (len(tokens) - (2 + (self.tokenizer != word_tokenize)) < 15):

            distribution = {}
            n_minus1_gram = tuple(tokens[-(self.ngram_size - 1):])

            for unigram in self.ngram_count[0]:
                if unigram not in [self.bos_token, 'UNK'] and self.ngram_count[0][unigram]:
                    distribution[unigram] = self.get_probability(n_minus1_gram, unigram, smoothing_args)

            unis = list(distribution.keys())
            vals = list(distribution.values())

            nextWord = random.choices(unis, weights=vals, k=1)
            tokens.append(nextWord[0])

            if tokens[-1] == self.eos_token:
                break

        if self.tokenizer == word_tokenize:
            print(' '.join(tokens))
        else:
            print(''.join(tokens))

        #################################################################################


def load_new_data(df):

    df_class1 = df.loc[df['funny'] > 1] # Funny
    df_class2 = df.loc[df['funny'] == 0] # Not funny

    class1_trn, class1_dev = train_test_split(df_class1['text'], test_size=0.2, random_state=42)
    class2_trn, class2_dev = train_test_split(df_class2['text'], test_size=0.2, random_state=42)

    class1_trn, class1_dev = class1_trn.reset_index(drop=True), class1_dev.reset_index(drop=True)
    class2_trn, class2_dev = class2_trn.reset_index(drop=True), class2_dev.reset_index(drop=True)

    display(df_class1[["text", 'funny']])
    display(df_class2[["text", 'funny']])

    return (class1_trn, class1_dev, class2_trn, class2_dev)

    #################################################################################

    
def predict_class(test_file, class1_lm, class2_lm, smoothing_args):


    textToPredict = open(test_file, 'r', encoding='utf-8').read()

    class1_ppl = class1_lm.get_perplexity(textToPredict, smoothing_args)
    class2_ppl = class2_lm.get_perplexity(textToPredict, smoothing_args)

    print(f"Perplexity for class1_lm: {class1_ppl}")
    print(f"Perplexity for class2_lm: {class2_ppl}")

    if class1_ppl < class2_ppl:
        print("It is in class 1")
    else:
        print("It is in class 2")

    #################################################################################
