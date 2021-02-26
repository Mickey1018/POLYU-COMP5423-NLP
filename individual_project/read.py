import numpy as np
# Read training data, validation data and test data form data set


def read_data():
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

    return train_X, train_y, val_X, val_y, test_X
