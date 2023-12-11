import numpy as np
from collections import defaultdict

class NB_Classifier:
    def __init__(self, alpha) -> None:
        self.alpha = alpha # Laplace smoothing parameter

        self.class_probabilities = {} # probability of review belonging to a class ie. prior probability
        self.feature_counts = {} # dict mapping label to sum of all features belonging to that label {1: [10, 20, 3...]}
        self.class_totals = {} # same as self.feature counts but array is summed. we this when calculating probability of a positive wordb being in a positive review
        self.vocab_size = 0 # total number of fetures


    """separated_by_class;
        Separates reviews by class and adds them to a dictionary

        Args:
            X (np.ndarray): Numpy array containing a sub array for every review each scores for ach feature 
            y (np.ndarray): array containing truth data

        Returns:
            dict: Dictionary mapping label to all reviews.
        """
    def separate_by_class(self, X, y):
        separated = {}
        for i in range(len(X)):
            sample, label = X[i], y[i]
            if label not in separated:
                separated[label] = []
            separated[label].append(sample)
        return separated
    
    """fit
        For each label it fills the dictionaries defined in __init__:
        self.class_totals is populated with the sum of all scores in positive / negative reviews
        self.class_probabilities: is populated with the prior probability for each label
        self.feature_counts: is populated with a vector containig the sum of each feature for each label

        Args:
            X (np.ndarray): Numpy array containing a sub array for every review each scores for ach feature 
            y (np.ndarray): array containing truth data
        """
    def fit(self, X, y):
        separated = self.separate_by_class(X, y)
        self.vocab_size = X.shape[1]
        
        for label, samples in separated.items():
            self.class_totals[label] = np.sum(samples)
            self.class_probabilities[label] = (len(samples) / len(X))
            
            self.feature_counts[label] = np.sum(samples, axis=0)


    """calculate_class_probabilities

        Args:
            sample (np.ndarray): review in vector fromata

        Returns:
            dict: mapping label to the sum of probabilities of each word being positive / negative
        """
    def calculate_class_probabilities(self, sample):
        probabilities = {}
        for label in self.class_probabilities:
            probabilities[label] = np.log(self.class_probabilities[label])
            for i, feature in enumerate(sample):
                if feature == 0:
                    continue
                # Calculate the log probability of the feature given the class using Laplace smoothing
                smoothed_feature_prob = np.log((self.feature_counts[label][i] + self.alpha) /
                                               (self.class_totals[label] + self.alpha * self.vocab_size))
                probabilities[label] += feature * smoothed_feature_prob
        return probabilities
    

    """predict

        Args:
            X (np.ndarray): vector containg vectors of input features

        Returns:
            np.ndarray: vector containing value 1 or 0 depending on the prediceted class
        """
    def predict(self, X):
        predictions = []
        for sample in X:
            probabilities = self.calculate_class_probabilities(sample)
            best_label = None
            best_prob = -np.inf
            for label, probability in probabilities.items():
                if probability > best_prob:
                    best_prob = probability
                    best_label = label
            predictions.append(best_label)
        return predictions