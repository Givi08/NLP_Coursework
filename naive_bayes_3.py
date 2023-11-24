import re
from collections import defaultdict

import pandas as pd
import numpy as np
import math

class NaiveBayes:
    def __init__(self):
        self.word_count = {}
        self.word_count_positive = {}
        self.word_count_negative = {}
        self.word_probability = {}

    def train(self, X_train, y_train):
        # X_train['y_train'] = y_train

        X_array = X_train.values
        for i in range(len(X_array)):
            for word in range(len(X_array[i])):
                if y_train[i] == 1 and X_array[i][word] !=0:   #Positive Review
                    if word in self.word_count_positive.keys():
                        self.word_count_positive[word]+=1
                    else:
                        self.word_count_positive[word] = 1
                elif y_train[i] == 0 and X_array[i][word] !=0: #Negative Review
                    if word in self.word_count_negative.keys():
                        self.word_count_negative[word]+=1
                    else:
                        self.word_count_negative[word] = 1
                if word in self.word_count.keys():        #For all words. I use this to compute probability.
                    self.word_count[word]+=1
                else:
                    self.word_count[word]=1

            

        # self.word_count_positive = sum(np.ceil(X_train[X_train.y_train == 1].values))
        # self.word_count_negative = sum(np.ceil(X_train[X_train.y_train == 0].values))
        # self.word_count = sum(np.ceil(X_train.values))

        # self.word_count_positive = dict(zip(self.word_count_positive, range(len(self.word_count_positive))))
        # self.word_count_negative = dict(zip(self.word_count_negative, range(len(self.word_count_negative))))
        # self.word_count = dict(zip(self.word_count, range(len(self.word_count))))

    def find_probability(self):
        for i in self.word_count:
          self.total_words =  self.word_count[i] 
        for i in self.word_count:
            self.word_probability[i] = (self.word_count[i] / self.total_words)
    
        threshold_p = (0.0001)
        for i in list(self.word_probability):
            if self.word_probability[i] < threshold_p:
                del self.word_probability[i]
                #self.word_probability = np.delete(self.word_probability, i)
                if i in range(len(self.word_count_positive)):   #list(dict) return it;s key elements
                    del self.word_count_positive[i]
                    #self. word_count_positive = np.delete(self.word_count_positive, i)
                if i in range(len(self.word_count_negative)):  
                    del self.word_count_negative[i]
                    #self.word_count_negative = np.delete(self.word_count_negative, i)
    
    def find_conditional_probability(self):
        self.total_positive_words = sum(self.word_count_positive.values())
        self.cp_positive = {}  #Conditional Probability
        for i in list(self.word_count_positive):
            self.cp_positive[i] = (self.word_count_positive[i] / self.total_positive_words)
        

        self.total_negative_words = sum(self.word_count_negative.values())
        self.cp_negative = {}  #Conditional Probability
        for i in list(self.word_count_negative):
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
        