import re
from collections import defaultdict

import pandas as pd
import numpy as np
import math

class NaiveBayes:
    def __init__(self):
        self.word_count = []
        self.word_count_positive = []
        self.word_count_negative = []
        self.word_probability = []

    def train(self, X_train, y_train):
        X_train['y_train'] = y_train

        self.word_count_positive = sum(np.ceil(X_train[X_train.y_train == 1].values))
        self.word_count_negative = sum(np.ceil(X_train[X_train.y_train == 0].values))
        self.word_count = sum(np.ceil(X_train.values))

        self.word_count_positive = dict(zip(self.word_count_positive, range(len(self.word_count_positive))))
        self.word_count_negative = dict(zip(self.word_count_negative, range(len(self.word_count_negative))))
        self.word_count = dict(zip(self.word_count, range(len(self.word_count))))

    def find_probability(self):
        self.word_probability = (np.array(list(self.word_count.values())) / sum(self.word_count.values()))
        
        threshold_p = (0.0001)
        # self.word_count = self.word_count[self.word_count>= threshold_p]
        # self.word_count_positive = self.word_count_positive[self.word_count_positive >= threshold_p]
        # self.word_count_negative = self.word_count_negative[self.word_count_negative >= threshold_p]


        for i in range(len(self.word_probability)):
            if self.word_probability[i] < threshold_p:
                self.word_probability = np.delete(self.word_probability, i)
                if i in range(len(self.word_count_positive)):   #list(dict) return it;s key elements
                    self. word_count_positive = np.delete(self.word_count_positive, i)
                if i in range(len(self.word_count_negative)):  
                    self.word_count_negative = np.delete(self.word_count_negative, i)
    
    def find_conditional_probability(self):
        self.total_positive_words = sum(self.word_count_positive)
        self.cp_positive = {}  #Conditional Probability
        for i in range(len(self.word_count_positive)):
            self.cp_positive[i] = (self.word_count_positive[i] / self.total_positive_words)
        

        self.total_negative_words = sum(self.word_count_negative)
        self.cp_negative = {}  #Conditional Probability
        for i in range(len(self.word_count_negative)):
            self.cp_negative[i] = (self.word_count_negative[i] / self.total_negative_words)


    def prediction(self, review):
        positive_term = 0.5
        negative_term = 0.5
            
        positive_M = len(self.cp_positive.keys())
        negative_M = len(self.cp_negative.keys())
        for word in range(len(review)):
            if word not in self.cp_negative.keys():
                negative_M +=1
            if word not in self.cp_positive.keys():
                positive_M += 1
            
        for word in range(len(review)):
            if word in self.cp_negative.keys():
                #print(negative_term)
                negative_term += math.log(self.cp_negative[word] + (1/negative_M))
                
            else:
                negative_term += math.log(1/negative_M)
            if word in self.cp_positive.keys():
                positive_term += math.log(self.cp_positive[word] + (1/positive_M))
            else:
                positive_term += math.log(1/positive_M)
        print(negative_term)
        print(positive_term)          
        if negative_term > positive_term: #negative_term/(negative_term + positive_term) > 0.5:
            return 0
        else:
            return 1
        