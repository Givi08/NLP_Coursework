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


    def train(self, review, label):
        # X_train.shape
        # for row in range(0,row_count):
        #     insincere += X_train.iloc[row]['target']
        #     sincere += (1 - X_train.iloc[row]['target'])
        #     sentence = X_train.iloc[row]['question_text']
        #     sentence = re.sub(r'\d+','',sentence)
        #     sentence = sentence.translate(sentence.maketrans("","",string.punctuation))
        #     words_in_sentence = list(set(sentence.split(' ')) - stop_words)
            
        #     for index,word in enumerate(words_in_sentence):
        #         word = stemmer.stem(word)
        #         words_in_sentence[index] = lemmatizer.lemmatize(word)
            
        for word in review: ## review aka word_in_sentence
            if label == 1:   #Positive Review
                if word in self.word_count_positive.keys():
                    self.word_count_positive[word]+=1
                else:
                    self.word_count_positive[word] = 1
            elif label == 0: #Negative Review
                if word in self.word_count_negative.keys():
                    self.word_count_negative[word]+=1
                else:
                    self.word_count_negative[word] = 1
            if word in self.word_count.keys():        #For all words. I use this to compute probability.
                self.word_count[word]+=1
            else:
                self.word_count[word]=1

        #print('Done')

    def find_probability(self):
        for i in self.word_count:
          self.total_words =  self.word_count[i] 
        for i in self.word_count:
            self.word_probability[i] = (self.word_count[i] / self.total_words)

        threshold_p = (0.00001)
        for i in list(self.word_probability):
            if self.word_probability[i] < threshold_p:
                del self.word_probability[i]
                if i in list(self.word_count_positive):   #list(dict) return it;s key elements
                    del self.word_count_positive[i]
                if i in list(self.word_count_negative):  
                    del self.word_count_negative[i]
        #print ('Total words ',len(self.word_probability))

    def find_conditional_probability(self):
        self.total_positive_words = sum(self.word_count_positive.values())
        self.cp_positive = {}  #Conditional Probability
        for i in list(self.word_count_positive):
            self.cp_positive[i] = math.log(self.word_count_positive[i] / self.total_positive_words)

        self.total_negative_words = sum(self.word_count_negative.values())
        self.cp_negative = {}  #Conditional Probability
        for i in list(self.word_count_negative):
            self.cp_negative[i] = math.log(self.word_count_negative[i] / self.total_negative_words)


    def prediction(self, review):
        ##
        # row_count = test.shape[0]

        # p_insincere = insincere / (sincere + insincere)
        # p_sincere = sincere / (sincere + insincere)
        # accuracy = 0

        # for row in range(0,row_count):
        #     sentence = test.iloc[row]['question_text']
        #     target = test.iloc[row]['target']
        #     sentence = re.sub(r'\d+','',sentence)
        #     sentence = sentence.translate(sentence.maketrans("","",string.punctuation))
        #     words_in_sentence = list(set(sentence.split(' ')) - stop_words)
        #     for index,word in enumerate(words_in_sentence):
        #         word = stemmer.stem(word)
        #         words_in_sentence[index] = lemmatizer.lemmatize(word)
            
            
        positive_term = 0.5
        negative_term = 0.5
            
        positive_M = len(self.cp_positive.keys())
        negative_M = len(self.cp_negative.keys())
        for word in review:
            if word not in self.cp_negative.keys():
                negative_M +=1
            if word not in self.cp_positive.keys():
                positive_M += 1
            
        for word in review:
            if word in self.cp_negative.keys():
                negative_term += math.log(self.cp_negative[word] + (1/negative_M))
            else:
                negative_term += math.log(1/negative_M)
            if word in self.cp_positive.keys():
                positive_term += math.log(self.cp_positive[word] + (1/positive_M))
            else:
                positive_term += math.log(1/positive_M)
                  
        if negative_term > positive_term: #negative_term/(negative_term + positive_term) > 0.5:
            return 'Negative'
        else:
            return 'Positive'
        
        if target == response:
            accuracy += 1
            
        print ('Accuracy is ',accuracy/row_count*100)

        ##

     
###############################################################################




    # def train(self, review, labels):
    #     for i in range(len(review)):
    #         # review = self.preprocess_text(reviews[i])
    #         words = review.split()
            
    #         if labels[i] == 'Positive':
    #             self.positive_reviews_count += 1
    #             for word in words:
    #                 self.positive_word_count[word] += 1
    #         elif labels[i] == 'Negative':
    #             self.negative_reviews_count += 1
    #             for word in words:
    #                 self.negative_word_count[word] += 1

    # def classify(self, review):
    #     # review = self.preprocess_text(review)
    #     words = review.split()
        
    #     # Calculate likelihoods for positive and negative classes
    #     positive_likelihood = 1.0
    #     negative_likelihood = 1.0
        
    #     for word in words:
    #         # Add Laplace smoothing to handle unseen words
    #         positive_likelihood *= (self.positive_word_count[word] + 1) / (self.positive_reviews_count + len(set(words)))
    #         negative_likelihood *= (self.negative_word_count[word] + 1) / (self.negative_reviews_count + len(set(words)))
        
    #     # Calculate probabilities using Bayes' theorem
    #     positive_probability = positive_likelihood * (self.positive_reviews_count / (self.positive_reviews_count + self.negative_reviews_count))
    #     negative_probability = negative_likelihood * (self.negative_reviews_count / (self.positive_reviews_count + self.negative_reviews_count))
        
    #     # Make the classification based on the higher probability
    #     if positive_probability > negative_probability:
    #         return 'Positive'
    #     else:
    #         return 'Negative'
