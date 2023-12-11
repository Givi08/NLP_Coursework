from sklearn.model_selection import train_test_split
import numpy as np

import nltk
import string

nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk import word_tokenize
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.probability import FreqDist


class PreProcessing:
    def __init__(self):
        # self.X_train = None
        # self.X_test = None
        # self.X_dev = None
        # self.y_train = None 
        # self.y_test = None
        # self.y_dev = None

        self.n_grams = 0 # number of n_grams
        self.stop_words = set(stopwords.words('english')) # Stopwords, this varribale is used by both stem and lemmatize

    """lowercase: sets all words to lowercase

    Args:
        text (list): list containing strings (reviews)

    Returns:
        list: list containig processed strings (reviews)
    """
    def lowercase(self, text):
        word_list =  [review.lower() for review in text]
        return word_list


    """remove_punctuation: removes all punctuation.

    Args:
        text (list): list containing strings (reviews)

    Returns:
        list: list containig processed strings (reviews)
    """
    def remove_punctuation(self, text):
        word_list = [''.join([char for char in review if char not in string.punctuation]) for review in text]
        return word_list
    
    
     
    """remove_stopwords: removes all stopwords.

    Args:
        text (list): list containing strings (reviews)

    Returns:
        list: list containig processed strings (reviews)
    """
    def remove_stopwords(self, text):
        word_list = [' '.join([word for word in word_tokenize(review) if word.lower() not in self.stop_words]) for review in text]
        return word_list
    
    
     
    """stem: using SnowballStemmer and nltk's word tokenizer performs stemming and tokenization.

    Args:
        text (list): list containing strings (reviews)

    Returns:
        list: list containig list of stemmed tokens for each review
    """
    def stem(self, text):
        st = SnowballStemmer('english')
        word_list = [[st.stem(word) for word in word_tokenize(review)] for review in text]
        return word_list
    

    
    """lemmatize: Using WordNetLemmatizer and nltk's word tokenizer performs stemming and tokenization.

    Args:
        text (list): list containing strings (reviews)

    Returns:
        list: list containig list of lemmatized tokens for each review
    """
    def lemmatize(self, text):
        lemmatizer = WordNetLemmatizer()
        word_list = [[lemmatizer.lemmatize(word) for word in word_tokenize(review)] for review in text]
        return word_list
    
    """zipfs_law: Using FreqDist calculates the most frequent words used and removes them.

    Args:
        percentage_removal (float): indicates the top percentage of wordsd being removed
        text (list): list of lists (one list per review) each containig  tokens.

    Returns:
        list: list containig the less popular tokens
    """
    def zipfs_law(self, percentage_removal, text):
        tokens = [item for sublist in text for item in sublist]
        freq_dist = FreqDist(tokens)
        ranked_words = sorted(freq_dist, key=freq_dist.get, reverse=True)
        to_del = set(ranked_words[int(percentage_removal * len(ranked_words)):])

        return [[word for word in sublist if word not in to_del] for sublist in text]

    """set_n_grams: sets both the number of n_grams and ouputs the original list processed so that each gram contains n tokens

    Args:
        n (int): Number of n_grams
        text (list): list of list containing tokens

    Returns:
        list: list of list with the processed tokens for n_grams
    """
    def set_n_grams(self, n, text):
        n_grams = []
        if n > 5:
            for review in text:
                n_grams.append(list(ngrams(review, 5)))
        else:
            for review in text:
                n_grams.append(list(ngrams(review, n)))
            
        self.n_grams = n
        return n_grams

    def get_n_grams(self):
        return self.n_grams
       
    """set_data_splits: returns the data split into 70% train, 15% test, 15% dev

    Args:
        X (np.ndarray): Features vector
        y (np.ndarray): truth data vecortor

    Returns:
        4 np.ndarray: containing the train, test, dev splits
    """
    def set_data_splits(self, X, y):
        
        # Split the dataset into training (70%), testing (15%), and development (15%)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_dev, y_test, y_dev = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        # self.X_train, self.X_test, self.X_dev, self.y_train, self.y_test, self.y_dev = X_train, X_test, X_dev, y_train, y_test, y_dev
        return X_train, X_test, X_dev, y_train, y_test, y_dev
            
            
    def remove_low_proba(self, features, threshold,):
        features = np.array(list(features.values()))

        proba_features = np.sum(features, axis = 0) / np.sum(features)

        sorted_indexes = np.argsort(proba_features)
        sorted_probs = np.sort(proba_features)

        cumulative_sum = np.cumsum(sorted_probs)

        remove_point = threshold * np.sum(proba_features)

        to_remove_sorted = np.argmax(cumulative_sum >= remove_point)

        indexes = list(sorted_indexes[to_remove_sorted:])


        return np.array([np.delete(row, indexes) for row in features])
