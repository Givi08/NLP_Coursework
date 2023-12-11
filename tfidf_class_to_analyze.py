from collections import defaultdict
import math

class TFIDF:
    def __init__(self):
        # Counters for positive and negative words
        self.positive_word_count = defaultdict(int)
        self.negative_word_count = defaultdict(int)

    """get_terms: maps each word to its count

    Args:
        processed_reviews (list): contains the preprocessed(tokenized, stopwords, lowercased...) words

    Returns:
        dict: dictionary mapping each review to a a dict of its word mapped to their counts in each document
    """
    def get_terms(self, processed_reviews):
        terms = {}
        for review in processed_reviews:
            terms[review] = (terms.get(review, 0) + 1) #/(len(processed_reviews))
        return terms
      
    """collect_vocabulary: gets a list every individual unique token.

    Args:
        review_terms (dict): dictionary outputted in get_terms

    Returns:
        list: contains each unique token
    """
    def collect_vocabulary(self, review_terms):
        all_terms = []
        for doc_id in review_terms.keys():
            for term in review_terms.get(doc_id).keys():
                all_terms.append(term)
        return sorted(set(all_terms))
    

    def vectorize(self, input_terms, shared_vocabulary):
        output = {}
        for item_id in input_terms.keys():
            terms = input_terms.get(item_id)
            output_vector = []
            for word in shared_vocabulary:
                if word in terms.keys():
                    output_vector.append(int(terms.get(word)))
                else:
                    output_vector.append(0)
            output[item_id] = output_vector
        return output
    
    """calculate_idfs: Calculates the idf for each term in the list returned from collect_vocabulary

    Args:
        shared_vocabulary (list): list returned from collect_vocabulary
        d_terms (dict): dictionary returned from get_terms

    Returns:
        dict: maps each word in shareed_vocabulary to the idf score
    """
    def calculate_idfs(self, shared_vocabulary, d_terms):
        doc_idfs = {}
        for term in shared_vocabulary:
            doc_count = 0
            for doc_id in d_terms.keys():
                terms = d_terms.get(doc_id)
                if term in terms.keys():
                    doc_count += 1

            ## divide term count by length of vector containing tokens
            doc_idfs[term] = math.log(float(len(d_terms.keys()))/float(1 + doc_count))
        return doc_idfs
    
    # def vectorize_idf(self, input_terms, input_idfs, shared_vocabulary):
    #     output = {}
    #     for item_id in input_terms.keys():
    #         terms = input_terms.get(item_id)
    #         output_vector = []
    #         for term in shared_vocabulary:
    #             if term in terms.keys():
    #                 output_vector.append(input_idfs.get(term)*float(terms.get(term)))
    #             else:
    #                 output_vector.append(float(0))
    #         output[item_id] = output_vector
    #     return output
    
    """vectorize_idf: returns a dictionary mapping each review to the normalized idf score of each word in the collective dictionary appearing in the reiview

    Args:
        input_terms (dict): dictionary returned from get_terms
        input_idfs (dict): dictionary returned from calculate_idfs
        shared_vocabulary (list): lisst returned from collect_vocabulary

    Returns:
        dict: dictionary mapping each review to the normalized idf score of each word in the collective dictionary appearing in the reiview
    """
    def vectorize_idf(self, input_terms, input_idfs, shared_vocabulary):
        output = {}
        
        for item_id, terms in input_terms.items():
            output_vector = [
                input_idfs.get(term, 0.0) * float(terms.get(term, 0.0))
                for term in shared_vocabulary
            ]
            output[item_id] = output_vector
        
        return output
