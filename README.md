# Natural Language Processing Coursework
Tasked with extracting features from IMDB reviews to classify Positive and Negative sentiment analysis.
Breakdown of tasks:
- Review extraction from files.
- Pre-processing using n-grams, lemmatization/stemming, lowercasing, and punctuation removal.
- Compute vector representation of features using TFIDF.
- Split into test/train/dev.
- Compute own implementation of Naive Bayes classifier.
- Compare with sklearn implementation of Naive Bayes, SGD, SVM and BERT.
- Use different pre-processing metrics to analyse.

My own implementation achieved identical results to sklearn's.


The results for my own Naive Bayes implementation are below:

| Lemmatization | Stemming | Zipf's law | Stopwords | Lowercase | Punctuation | Accuracy | Precision | Recall | F1 Score |
|---------------|----------|------------|-----------|-----------|-------------|----------|-----------|--------|----------|
| TRUE          | FALSE    | 2%         | TRUE      | FALSE     | TRUE        | 82.0%    | 82.2%     | 81.6%  | 81.9%    |
| TRUE          | FALSE    | 2%         | TRUE      | FALSE     | FALSE       | 81.7%    | 82.3%     | 80.6%  | 81.4%    |
| TRUE          | FALSE    | 2%         | TRUE      | TRUE      | TRUE        | 81.5%    | 81.3%     | 81.6%  | 81.5%    |
| FALSE         | TRUE     | 2%         | FALSE     | TRUE      | FALSE       | 80.7%    | 81.2%     | 79.6%  | 80.4%    |
| TRUE          | FALSE    | 2%         | FALSE     | FALSE     | TRUE        | 80.5%    | 79.9%     | 81.3%  | 80.6%    |
| TRUE          | FALSE    | 0%         | TRUE      | FALSE     | TRUE        | 80.5%    | 81.4%     | 78.9%  | 80.1%    |
| FALSE         | TRUE     | 2%         | FALSE     | FALSE     | FALSE       | 80.5%    | 81.0%     | 79.6%  | 80.3%    |
| TRUE          | FALSE    | 2%         | TRUE      | TRUE      | FALSE       | 80.3%    | 79.7%     | 81.3%  | 80.5%    |
| TRUE          | FALSE    | 2%         | FALSE     | TRUE      | TRUE        | 80.0%    | 79.9%     | 79.9%  | 79.9%    |
| TRUE          | FALSE    | 2%         | FALSE     | TRUE      | FALSE       | 80.0%    | 80.1%     | 79.6%  | 79.9%    |
| FALSE         | TRUE     | 2%         | TRUE      | TRUE      | FALSE       | 80.0%    | 79.7%     | 80.3%  | 80.0%    |
| TRUE          | FALSE    | 0%         | FALSE     | FALSE     | TRUE        | 79.8%    | 80.9%     | 77.9%  | 79.4%    |
| FALSE         | TRUE     | 0%         | TRUE      | TRUE      | TRUE        | 79.8%    | 83.0%     | 74.9%  | 78.7%    |
| FALSE         | TRUE     | 0%         | TRUE      | FALSE     | TRUE        | 79.8%    | 83.0%     | 74.9%  | 78.7%    |
| FALSE         | TRUE     | 0%         | FALSE     | TRUE      | TRUE        | 79.8%    | 83.5%     | 74.3%  | 78.6%    |
| FALSE         | TRUE     | 0%         | FALSE     | FALSE     | TRUE        | 79.8%    | 83.5%     | 74.3%  | 78.6%    |
| FALSE         | TRUE     | 2%         | TRUE      | FALSE     | FALSE       | 79.7%    | 79.4%     | 79.9%  | 79.7%    |
| TRUE          | FALSE    | 0%         | TRUE      | TRUE      | TRUE        | 79.7%    | 81.3%     | 76.9%  | 79.0%    |
| TRUE          | FALSE    | 2%         | FALSE     | FALSE     | FALSE       | 79.5%    | 79.5%     | 79.3%  | 79.4%    |
| TRUE          | FALSE    | 0%         | TRUE      | FALSE     | FALSE       | 79.2%    | 80.6%     | 76.6%  | 78.6%    |
| FALSE         | TRUE     | 2%         | TRUE      | TRUE      | TRUE        | 79.2%    | 78.8%     | 79.6%  | 79.2%    |
| FALSE         | TRUE     | 2%         | TRUE      | FALSE     | TRUE        | 79.2%    | 78.8%     | 79.6%  | 79.2%    |
| TRUE          | FALSE    | 0%         | FALSE     | TRUE      | FALSE       | 79.0%    | 81.2%     | 75.3%  | 78.1%    |
| FALSE         | TRUE     | 0%         | FALSE     | TRUE      | FALSE       | 78.7%    | 82.3%     | 72.9%  | 77.3%    |
| TRUE          | FALSE    | 0%         | FALSE     | TRUE      | TRUE        | 78.5%    | 80.4%     | 75.3%  | 77.7%    |
| FALSE         | TRUE     | 2%         | FALSE     | TRUE      | TRUE        | 78.5%    | 78.2%     | 78.9%  | 78.5%    |
| FALSE         | TRUE     | 2%         | FALSE     | FALSE     | TRUE        | 78.5%    | 78.2%     | 78.9%  | 78.5%    |
| FALSE         | TRUE     | 0%         | TRUE      | TRUE      | FALSE       | 78.5%    | 81.5%     | 73.6%  | 77.3%    |
| FALSE         | TRUE     | 0%         | TRUE      | FALSE     | FALSE       | 78.2%    | 81.1%     | 73.2%  | 77.0%    |
| TRUE          | FALSE    | 0%         | TRUE      | TRUE      | FALSE       | 78.2%    | 80.0%     | 74.9%  | 77.4%    |

More results can be achieved by running through the Notebook `Final_submission.ipynb`




TODO
- Clean up code.
- Remove useless files.
- Optimize running time. Simplify TFIDF.
