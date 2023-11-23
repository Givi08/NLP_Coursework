import numpy as np
from collections import defaultdict

class GaussianNaiveBayes:
    def __init__(self):
        self.summaries = {}
    
    def separate_by_class(self, X, y):
        separated = defaultdict(list)
        for i in range(len(X)):
            separated[y[i]].append(X[i])
        return separated
    
    def calculate_statistics(self, features):
        mean = np.mean(features, axis=0)
        std_dev = np.std(features, axis=0)
        return mean, std_dev
    
    def summarize_by_class(self, X, y):
        separated = self.separate_by_class(X, y)
        summaries = {}
        for class_value, instances in separated.items():
            summaries[class_value] = self.calculate_statistics(instances)
        return summaries
    
    def calculate_probability(self, x, mean, stdev):
        exponent = np.exp(-((x - mean) ** 2) / (2 * stdev ** 2))
        return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent
    
    def calculate_class_probabilities(self, summaries, input_data):
        probabilities = {}
        for class_value, class_summaries in summaries.items():
            probabilities[class_value] = 1
            for i in range(len(class_summaries[0])):
                mean, std_dev = class_summaries[0][i], class_summaries[1][i]
                x = input_data[i]
                probabilities[class_value] *= self.calculate_probability(x, mean, std_dev)
        return probabilities
    
    def train(self, X, y):
        self.summaries = self.summarize_by_class(X, y)
    
    def predict(self, sample):
        probabilities = self.calculate_class_probabilities(self.summaries, sample)
        predicted_class = max(probabilities, key=probabilities.get)
        return predicted_class

# Sample data
data = {
    'Review': [
        [0.2, 0.8, 0.5, 0.7],  # Feature values for the first review
        [0.6, 0.1, 0.4, 0.9],  # Feature values for the second review
        [0.9, 0.2, 0.7, 0.8],  # Feature values for the third review
        [0.1, 0.7, 0.3, 0.6]   # Feature values for the fourth review
    ],
    'Sentiment': ['Positive', 'Negative', 'Positive', 'Negative']
}

# Convert sentiment labels to numerical values
sentiment_map = {'Positive': 1, 'Negative': 0}
data['Sentiment'] = [sentiment_map[s] for s in data['Sentiment']]

# Separate features and labels
X_train = np.array(data['Review'])
y_train = np.array(data['Sentiment'])

# Create an instance of the Gaussian Naive Bayes classifier
gnb = GaussianNaiveBayes()

# Train the model
gnb.train(X_train, y_train)

# Test data
test_data = [0.4, 0.3, 0.6, 0.5]  # Test data point
predicted_class = gnb.predict(test_data)
predicted_sentiment = 'Positive' if predicted_class == 1 else 'Negative'

print(f"Predicted Sentiment: {predicted_sentiment}")
