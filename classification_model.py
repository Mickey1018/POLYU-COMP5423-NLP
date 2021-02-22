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
from sklearn.neural_network import MLPClassifier


# Building Machine Learning Model
def train_model(classifier, feature_maps, training_label):
    clf = None
    if classifier == 'logistic regression':
        clf = make_pipeline(StandardScaler(),
                            SGDClassifier(max_iter=1000, tol=1e-3, loss="log", penalty="l2"))
    elif classifier == 'svm':
        tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                             'C': [1, 10, 100, 1000]},
                            {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        clf = make_pipeline(StandardScaler(),
                            GridSearchCV(SVC(gamma='auto'), param_grid=tuned_parameters))
    elif classifier == 'naive bayes':
        clf = make_pipeline(StandardScaler(),
                            GaussianNB())
    elif classifier == 'random forest':
        clf = make_pipeline(StandardScaler(),
                            RandomForestClassifier(class_weight="balanced_subsample"))
    elif classifier == 'mlp':
        clf = make_pipeline(StandardScaler(),
                            MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=200))
    clf.fit(feature_maps, training_label)
    return clf


# Build function to predict labels with trained model
def predict_emotion(trained_model, feature_maps):
    return trained_model.predict(feature_maps)


# Model Evaluation - Accuracy, F1-score, Area under curve
def evaluation(true_label, predicted_label):
    # print("accuracy: ", metrics.accuracy_score(true_label, predicted_label))
    # print("f1 score: ", metrics.f1_score(true_label, predicted_label, average=None))
    # print("area under roc curve: ", metrics.roc_auc_score(true_label, predicted_label))
    print("classification report: \n", metrics.classification_report(true_label, predicted_label))
    return True
