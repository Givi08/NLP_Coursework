from sklearn.model_selection import train_test_split
import pandas as pd

import nltk
import string

nltk.download('stopwords')
nltk.download('punkt')

from nltk import word_tokenize
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

import operator

class PreProcessing:
    def __init__(self):
        # Counters for positive and negative words
        self.X_train = None
        self.X_test = None
        self.X_dev = None
        self.y_train = None 
        self.y_test = None
        self.y_dev = None

    

    def stemming(self, text, include_stopwords = False, include_punctuation = False, keep_uppercase = False):
        ## Use Stopwords example from labs
        stoplist = set(stopwords.words('english'))
        st = LancasterStemmer()

        if not keep_uppercase:
            text = text.lower()

        if not include_stopwords and not include_punctuation:
            word_list = [st.stem(word) for word in word_tokenize(text)
                if (not word in stoplist) and (not word in string.punctuation)]
        
        elif not include_stopwords and include_punctuation:
            word_list = [st.stem(word) for word in word_tokenize(text)
                if (not word in stoplist)]
        
        elif include_stopwords and not include_punctuation:
            word_list = [st.stem(word) for word in word_tokenize(text)
                if (not word in string.punctuation)]
            
        else:
            word_list = [st.stem(word) for word in word_tokenize(text)]

            
        return word_list
    
    def set_n_grams(self, n, text):
        if n > 5:
            n_grams = list(ngrams(text, 5))
        else:
            n_grams = list(ngrams(text, n))
        self.n_grams = n_grams
        return n_grams
       

    def freq_dist(word_list, up_to):
        word_map = {}
        for a_word in word_list:
            word_map[a_word] = word_map.get(a_word, 0) + 1
        
        total_count = sum(word_map.values())
        sorted_map = (sorted(word_map.items(), key=operator.itemgetter(1)))[::-1]
        percentage_map = [(item[0], 100*float(item[1])/float(total_count)) for item in sorted_map[:up_to]]
        return percentage_map


    def tfidf(self, text):
        ## Use TFIDF example from labs
        ## Document here is the class Positive or Negative Review
        terms = {}
        for word in text:
            terms[word] = terms.get(word, 0) + 1
        return terms


    def set_data_splits(self, data):
        ## Split pandas dataframe and store how it's being split (3 splits)
        X = data.review
        y = data.label

        # Split the dataset into training (70%), testing (15%), and development (15%)
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
        X_test, X_dev, y_test, y_dev = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        self.X_train, self.X_test, self.X_dev, self.y_train, self.y_test, self.y_dev = X_train, X_test, X_dev, y_train, y_test, y_dev

        return X_train, X_test, X_dev, y_train, y_test, y_dev
            
            
    
    

    # def preprocess_text(self, text):
    #     # Simple text preprocessing: lowercase and remove non-alphanumeric characters
    #     text = text.lower()
    #     text = re.sub(r'[^a-zA-Z\s]', '', text)
    #     return text
