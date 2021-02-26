import os
import pickle
from read import *
from write import *
from text_processor import *
from feature_extraction import *
from classification_model import *


def main():

    # Read training data, validation data and test data form data set
    train_X, train_y, val_X, val_y, test_X = read_data()

    # Process training data, validation data and test data
    processed_train_X = text_processing(train_X)
    processed_val_X = text_processing(val_X)
    processed_test_X = text_processing(test_X)

    # extract features from training data, validation data and test data
    features_train = extract_features(processed_train_X)
    features_val = extract_features(processed_val_X)
    features_test = extract_features(processed_test_X)

    # Train a machine learning model 'MLP' with training data set
    model = train_model(classifier='random forest',
                        feature_maps=features_train,
                        training_label=train_y)

    # save the trained machine learning model
    if os.path.exists('trained_model.sav'):
        os.remove('trained_model.sav')
    pickle.dump(model, open('trained_model.sav', 'wb'))

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

    # write  result into text file
    write_data(predicted_test_y)


if __name__ == "__main__":
    main()
