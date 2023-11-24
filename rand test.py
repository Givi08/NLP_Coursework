import numpy as np
from collections import defaultdict

class NB_Classifier:
    def __init__(self) -> None:
        pass
    
    def fit(self, X, y):
        self.class_probabilities = np.zeros(len(X.label.nunique()))
        self.cond_proba = np.zeros(shape=(len(X.label.nunique(), len(X))))
        for i in range(X.label.nunique()):
            self.class_probabilities[i] = len(X[X.label == i])/(len(X))
            
            tf = np.array(X[X.label == i].sum())

            self.cond_proba[i, :] = ( (tf + 1) / (np.sum(tf) + 1 * len(X)))


    
    
    
    def calculate_probability(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))

        return ((1 / self.value * std) * exponent)
    

    


    def calculate_class_probabilities(self, sample):
        probabilities = {}
        for label, class_prob in self.class_probabilities.items():
            probabilities[label] = class_prob
            for i , feature in enumerate(sample):
                mean, std_dev = self.feature_stats[label]['mean'][i], self.feature_stats[label]['std'][i]
                probabilities[label] *= self.calculate_probability(feature, mean, std_dev)
        return probabilities
    
    def predict(self, X):
        predictions = []
        for sample in X:
            probabilities = self.calculate_class_probabilities(sample)
            best_label = None
            best_prob = -1
            for label, probability in probabilities.items():
                if probability > best_prob:
                    best_prob = probability
                    best_label = label
            predictions.append(best_label)
        return predictions


