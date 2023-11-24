from collections import defaultdict
import pandas as pd
import nltk
import math

class TFIDF:
    def __init__(self):
        # Counters for positive and negative words
        self.positive_word_count = defaultdict(int)
        self.negative_word_count = defaultdict(int)

    def get_terms(self, processed_reviews):
        terms = {}
        for review in processed_reviews:
            terms[review] = (terms.get(review, 0) + 1) #/(len(processed_reviews))
        return terms
      
    
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
    
    def vectorize_idf(self, input_terms, input_idfs, shared_vocabulary):
        output = {}
        for item_id in input_terms.keys():
            terms = input_terms.get(item_id)
            output_vector = []
            for term in shared_vocabulary:
                if term in terms.keys():
                    output_vector.append(input_idfs.get(term)*float(terms.get(term)))
                else:
                    output_vector.append(float(0))
            output[item_id] = output_vector
        return output





###########################################################

# tfidf = TFIDF()

# reviews = pd.read_csv('reviews.csv')
# print('here')
# review_terms = {}
# label_terms = {}
# for review_id in reviews.index:
#     review_terms[review_id] = tfidf.get_terms(reviews.review.loc[review_id])
# for label_id in reviews.index:
#     label_terms[label_id] = tfidf.get_terms(reviews.label.loc[label_id])


# print(f"{len(review_terms)} documents in total")
# d1_terms = review_terms.get(1)
# print("Terms and frequencies for document with id 1:")
# print(d1_terms)
# print(f"{len(d1_terms)} terms in this document")
# print()
# print(f"{len(label_terms)} queries in total")
# q1_terms = label_terms.get(1)
# print("Terms and frequencies for query with id 1:")
# print(q1_terms)
# print(f"{len(q1_terms)} terms in this query")


# all_terms = tfidf.collect_vocabulary(review_terms, label_terms)
# print(f"{len(all_terms)} terms in the shared vocabulary")
# print("First 10:")
# print(all_terms[:10])


# doc_vectors = tfidf.vectorize(review_terms, all_terms)
# qry_vectors = tfidf.vectorize(label_terms, all_terms)

# print(f"{len(doc_vectors)} document vectors")
# d1460_vector = doc_vectors.get(1460)
# print(f"{len(d1460_vector)} terms in this document")
# print(f"{len(qry_vectors)} query vectors")
# q112_vector = qry_vectors.get(112)
# print(f"{len(q112_vector)} terms in this query")


# doc_idfs = tfidf.calculate_idfs(all_terms, review_terms)
# print(f"{len(doc_idfs)} terms with idf scores")
# print("Idf score for the word system:")
# print(doc_idfs.get("system"))

# doc_vectors = tfidf.vectorize_idf(review_terms, doc_idfs, all_terms)

# print(f"{len(doc_vectors)} document vectors")
# print("Number of idf-scored words in a particular document:")
# print(len(doc_vectors.get(1460)))

