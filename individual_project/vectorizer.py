from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from read import *
from text_processor import *
import os
import pickle

# Learn the vocabulary dictionary and return document-term matrix.

# Create a vectorizer to convert a collection of text documents to a matrix of token counts
count_vectorizer = CountVectorizer(analyzer="word",  # make sure features are made of word n-gram
                                   preprocessor=None,
                                   stop_words=None,
                                   max_features=10,  # 10 features that occur the most frequently across the corpus
                                   ngram_range=(1, 1)  # extract unigram
                                   )

tfidf_vectorizer = TfidfVectorizer(analyzer="word",  # make sure features are made of word n-gram
                                   preprocessor=None,
                                   stop_words=None,
                                   max_features=6000,  # 6000 features that occur the most frequently across the corpus
                                   ngram_range=(1, 2)  # extract unigram, bigram
                                   )

# train TFIDF vectorizer
train_X, train_y, val_X, val_y, test_X = read_data()
processed_train_X = text_processing(train_X)
tfidf_vectorizer.fit_transform(processed_train_X)

# save the trained vectorizer into disk
if os.path.exists('trained_vectorizer.sav'):
    os.remove('trained_vectorizer.sav')
pickle.dump(tfidf_vectorizer, open('trained_vectorizer.sav', 'wb'))


