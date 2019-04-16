import numpy as np
import pandas as pd
import time
import math
import collections

class naive_bayes:
    len_unique_term = 0
    genres = []
    len_train = 0
    class_prob = {}
    word_count = {}
    total_word = {}

    def initialize(self,x_train, y_train, genres):
        self.genres = genres
        self.len_unique_term = self.count_unique_term(x_train)
        self.len_train = x_train.shape[0]
        for genre in genres:
            self.class_prob[genre] = math.log(self.count_total_sample_class(x_train,y_train, genres.index(genre))/self.len_train)
            self.total_word[genre] = self.count_total_word_class(x_train, y_train, genres.index(genre))
            self.word_count[genre] = self.count_word_occurance_class(x_train, y_train, genres.index(genre))

    def predict(self,x_test):
        score = {}
        for genre in self.genres:
            score[genre] = self.class_prob[genre]
            for term in x_test:
                counter = self.word_count[genre]
                if term in counter:
                    score[genre] += math.log(self.word_count[genre][term]+1/self.total_word[genre])
                else:
                    score[genre] += math.log(1/self.total_word[genre])
        return score

    def count_unique_term(self, x_train):
        start = time.clock()
        words = []
        for x in x_train:
            words.extend(x)
        vocab_len = len(words)
        return vocab_len

    def count_total_sample_class(self, x_train, y_train, class_index):
        start = time.clock()
        total = np.sum(y_train[:,class_index]==1)
        return total
    
    def count_total_word_class(self, x_train, y_train, class_index):
        start = time.clock()
        x_class = x_train[y_train[:,class_index]==1]
        total_words = 0
        for x in x_class:
            total_words += len(x)
        return total_words        
    
    def count_word_occurance_class(self, x_train, y_train, class_index):
        start = time.clock()
        x_class = x_train[y_train[:,class_index]==1]
        counter = collections.Counter(x for xs in x_class for x in set(xs
        return counter
    
