from sklearn.model_selection import train_test_split
import pandas as pd

import nltk
import string

nltk.download('stopwords')
nltk.download('punkt')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

class PreProcessing:
    def __init__(self):
        # Counters for positive and negative words
        self.positive_word_count = defaultdict(int)
        self.negative_word_count = defaultdict(int)
        
        # Counters for positive and negative reviews
        self.positive_reviews_count = 0
        self.negative_reviews_count = 0

    def set_n_grams(self, n_grams):
        if n_grams > max_n_grams:
            self.n_grams = max_n_grams
        else:
            self.n_grams = n_grams

    def stopwords(self, text):
        ## Use Stopwords example from labs
        stoplist = set(stopwords.words('english'))
        st = LancasterStemmer()
        word_list = [st.stem(word) for word in word_tokenize(text.lower())
                 if not word in stoplist and not word in string.punctuation]
        return word_list

    def lemmatization(self, text):
        ## Use lemmatization example from labs
        return 0

    def stemming (self, text):
        ## Use stemming example from labs
        return 0

    def tfidf(self, text):
        ## Use TFIDF example from labs
        return 0

    def set_data_splits(self, data):
        ## Split pandas dataframe and store how it's being split (3 splits)
        X = data.Reviews
        y = data.labels

        # Split the dataset into training (70%), testing (15%), and development (15%)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_dev, y_test, y_dev = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

            
            

    

    # def preprocess_text(self, text):
    #     # Simple text preprocessing: lowercase and remove non-alphanumeric characters
    #     text = text.lower()
    #     text = re.sub(r'[^a-zA-Z\s]', '', text)
    #     return text
