import string
import nltk
import numpy as np
import sklearn
import gensim
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import os
from text_processor import *
from feature_extraction import *
from classification_model import *


def main():
    # Read training data, validation data and test data form data set

    # initialize a list to store training labels
    train_y = []
    # initialize a list to store training inputs
    train_X = []

    with open('data/train.txt') as train_data_set:
        sentences = train_data_set.readlines()
        for sentence in sentences:
            # split sentence to get training label
            splitted_sentence = sentence.split(';')
            # append the training inputs to list
            train_X.append(splitted_sentence[0].strip().lower())
            # append the training labels to list
            train_y.append(splitted_sentence[1].strip().lower())

    # initialize a list to store validation labels
    val_y = []
    # initialize a list to store validation inputs
    val_X = []

    with open('data/val.txt') as val_data_set:
        sentences = val_data_set.readlines()
        for sentence in sentences:
            # split sentence to get training label
            splitted_sentence = sentence.split(';')
            # append the validation inputs to list
            val_X.append(splitted_sentence[0].strip().lower())
            # append the validation labels to list
            val_y.append(splitted_sentence[1].strip().lower())

    # initialize a list to store test inputs
    test_X = []

    with open('data/test_data.txt') as test_data_set:
        sentences = test_data_set.readlines()
        for sentence in sentences:
            # append the test inputs to list
            test_X.append(sentence.strip().lower())

    # build function to convert label to numeric value
    def label2numeric(_labels):
        # mapping emotion to numeric value
        label_mapping = {'anger': -3, 'fear': -2, 'joy': 1, 'love': 2, 'sadness': -1, 'surprise': 3}
        y_index = []
        for label in _labels:
            y_index.append(label_mapping[label])
        return np.array(y_index)

    # convert training label to numeric value
    train_y = label2numeric(train_y)
    # convert validation label to numeric value
    val_y = label2numeric(val_y)

    # Process training data, validation data and test data
    processed_train_X = text_processing(train_X)
    processed_val_X = text_processing(val_X)
    processed_test_X = text_processing(test_X)

    # extract features from training data, validation data and test data
    features_train = extract_features(processed_train_X)
    features_val = extract_features(processed_val_X)
    features_test = extract_features(processed_test_X)

    # Train a machine learning model 'MLP' with training data set
    model = train_model(classifier='mlp',
                        feature_maps=features_train,
                        training_label=train_y)

    # Evaluate model with validation data set
    predicted_val_y = predict_emotion(trained_model=model,
                                      feature_maps=features_val)
    evaluation(val_y, predicted_val_y)

    # Predict labels for test data set and write in test_prediction.txt
    predicted_test_y_index = predict_emotion(trained_model=model,
                                             feature_maps=features_test)

    # convert index into string
    predicted_test_y = []
    for y in predicted_test_y_index:
        label_mapping = {'anger': -3, 'fear': -2, 'joy': 1, 'love': 2, 'sadness': -1, 'surprise': 3}
        for key, value in label_mapping.items():
            if y == value:
                predicted_test_y.append(key)

    # define directory to store file
    filePath = 'data/test_prediction.txt'

    # delete file if exist
    if os.path.exists(filePath):
        os.remove(filePath)

    # write file
    file = open('data/test_prediction.txt', 'w+')
    for result in predicted_test_y:
        file.write(result+'\n')
    file.close()


if __name__ == "__main__":
    main()
