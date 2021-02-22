from sklearn.feature_extraction.text import CountVectorizer

# Create a vectorizer to convert a collection of text documents to a matrix of token counts
vectorizer = CountVectorizer(analyzer="word",  # make sure features are made of word n-gram
                             preprocessor=None,
                             stop_words=None,
                             max_features=10,  # 10 features that occur the most frequently across the corpus
                             ngram_range=(1, 1)  # extract unigram
                             )
