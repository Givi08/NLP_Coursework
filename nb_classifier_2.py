import numpy as np
from collections import defaultdict

class NB_Classifier:
    def __init__(self) -> None:
        pass

    # Naive Bayes classifier functions
    # def separate_by_class(X, y):
    #     separated = defaultdict(list)
    #     for i in range(len(X)):
    #         separated[y[i]].append(X[i])
    #     return separated

    # def calculate_statistics(features):
    #     mean = np.mean(features, axis=0)
    #     std_dev = np.std(features, axis=0)
    #     return mean, std_dev

    def summarize_by_class(self, X, y):
        separated = defaultdict(list)
        for i in range(len(X)):
            separated[y.iloc[i]].append(X[i])

        summaries = {}
        for class_value, instances in separated.items():
            mean = np.mean(instances, axis=0)
            std_dev = np.std(instances, axis=0)
            summaries[class_value] = (mean, std_dev)
        return summaries

    # def calculate_probability(x, mean, stdev):
    #     exponent = np.exp(-((x - mean) ** 2) / (2 * stdev ** 2))
    #     return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

    def calculate_class_probabilities(self, summaries, input_data):
        probabilities = {}
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = 1
            for i in range(len(class_summaries[0])):
                mean, std_dev = class_summaries[0][i], class_summaries[1][i]
                x = input_data[i]
                exponent = np.exp(-((x - mean) ** 2) / (2 * std_dev ** 2))
                calculated_probability = (1 / (np.sqrt(2 * np.pi) * std_dev)) * exponent
                probabilities[class_value] *= calculated_probability
        return probabilities


