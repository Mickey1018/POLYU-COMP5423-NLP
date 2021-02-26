import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
import pickle
import nltk


# Build function to extract features
def extract_features(corpus):
    """
    A function to extract features from the input texts
    :param corpus: texts for feature extraction
    :return: features of texts
    """
    # 1. word count
    def word_count(sentence):
        words = []
        for word in sentence.split(' '):
            words.append(word)
        return len(words)
    word_count_feature = [word_count(x) for x in corpus]

    # 2. tf-idf weight
    tfidf = pickle.load(open('trained_vectorizer.sav', 'rb'))
    tfidf_feature = tfidf.transform(corpus).toarray()

    # 3. positive or negative
    # use NLTK built-in pretrained sentiment analyzer VADER
    sia = SentimentIntensityAnalyzer()

    def is_positive(sentence):
        if sia.polarity_scores(sentence)["compound"] > 0:
            return 1
        else:
            return -1
    polarity_feature = [is_positive(x) for x in corpus]

    # 4. polarity score
    # use NLTK built-in pretrained sentiment analyzer VADER
    sia = SentimentIntensityAnalyzer()

    def get_polarity_score(sentence):
        return sia.polarity_scores(sentence)["compound"]

    polarity_score_feature = [get_polarity_score(x) for x in corpus]

    # 5,6. count words start with 'un' and 'dis'
    def find_prefix(sentence, prefix):
        counter = 0
        for word in sentence.split():
            if word.startswith(prefix):
                counter -= 1
        return counter
    un_feature = [find_prefix(x, 'un') for x in corpus]
    dis_feature = [find_prefix(x, 'dis') for x in corpus]

    # 7. find words related to different emotions in sentence
    def find_key_words(sentence):

        synonyms_emotion = \
            {
             'anger': {
                 'synonyms:': ['angry', 'choler', 'furor', 'wrath', 'madness', 'indignation', 'irateness'],
                 'value': -3},
             'fear': {
                 'synonyms': ['fear', 'anxiety', 'dread', 'fright', 'horror', 'panic', 'scare'],
                 'value': -2},
             'joy': {
                 'synonyms': ['joy', 'beatitude', 'blessedness', 'bliss', 'felicity', 'gladness', 'happiness'],
                 'value': 1},
             'love': {
                 'synonyms': ['love', 'appreciate', 'cherish', 'prize', 'treasure', 'value'],
                 'value': 2},
             'sadness': {
                 'synonyms': ['sad', 'blues', 'depression', 'gloom', 'dreariness', 'despond', 'dejection'],
                 'value': -1},
             'surprise': {
                 'synonyms': ['surprise', 'bombshell', 'jar', 'jolt', 'stunner', 'shock'],
                 'value': 3}
             }
        wln = nltk.WordNetLemmatizer()
        for emotion, att in synonyms_emotion.items():
            for key, value in att.items():
                if not isinstance(value, int):
                    for word in value:
                        wln.lemmatize(word)

        def count(sentence):
            count_key_words = {'anger': 0, 'fear': 0, 'joy': 0, 'love': 0, 'sadness': 0, 'surprise': 0}
            for word in sentence.split():
                for emotion, att in synonyms_emotion.items():
                    for key, value in att.items():
                        if not isinstance(value, int):
                            if word in value:
                                count_key_words[emotion] += 1
                        else:
                            continue
            return count_key_words

        counted = count(sentence)

        max = 0
        emotion = None

        for key_word, count in counted.items():
            if max < count:
                max = count
                emotion = key_word
        if not emotion:
            return 0
        else:
            return synonyms_emotion[emotion]['value']

    emotion_feature = [find_key_words(x) for x in corpus]

    # concat different features
    all_features = np.concatenate((tfidf_feature, np.array([polarity_feature]).T), axis=1)
    all_features = np.concatenate((all_features, np.array([polarity_score_feature]).T), axis=1)
    all_features = np.concatenate((all_features, np.array([word_count_feature]).T), axis=1)
    all_features = np.concatenate((all_features, np.array([un_feature]).T), axis=1)
    all_features = np.concatenate((all_features, np.array([dis_feature]).T), axis=1)
    all_features = np.concatenate((all_features, np.array([emotion_feature]).T), axis=1)

    return all_features
