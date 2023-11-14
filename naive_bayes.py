import re
from collections import defaultdict

import pandas as pd

class NaiveBayes:
    def __init__(self):
        # Counters for positive and negative words
        self.positive_word_count = defaultdict(int)
        self.negative_word_count = defaultdict(int)
        
        # Counters for positive and negative reviews
        self.positive_reviews_count = 0
        self.negative_reviews_count = 0

    # def preprocess_text(self, text):
    #     # Simple text preprocessing: lowercase and remove non-alphanumeric characters
    #     text = text.lower()
    #     text = re.sub(r'[^a-zA-Z\s]', '', text)
    #     return text

    def train(self, review, labels):
        for i in range(len(reviews)):
            # review = self.preprocess_text(reviews[i])
            words = review.split()
            
            if labels[i] == 'Positive':
                self.positive_reviews_count += 1
                for word in words:
                    self.positive_word_count[word] += 1
            elif labels[i] == 'Negative':
                self.negative_reviews_count += 1
                for word in words:
                    self.negative_word_count[word] += 1

    def classify(self, review):
        # review = self.preprocess_text(review)
        words = review.split()
        
        # Calculate likelihoods for positive and negative classes
        positive_likelihood = 1.0
        negative_likelihood = 1.0
        
        for word in words:
            # Add Laplace smoothing to handle unseen words
            positive_likelihood *= (self.positive_word_count[word] + 1) / (self.positive_reviews_count + len(set(words)))
            negative_likelihood *= (self.negative_word_count[word] + 1) / (self.negative_reviews_count + len(set(words)))
        
        # Calculate probabilities using Bayes' theorem
        positive_probability = positive_likelihood * (self.positive_reviews_count / (self.positive_reviews_count + self.negative_reviews_count))
        negative_probability = negative_likelihood * (self.negative_reviews_count / (self.positive_reviews_count + self.negative_reviews_count))
        
        # Make the classification based on the higher probability
        if positive_probability > negative_probability:
            return 'Positive'
        else:
            return 'Negative'
