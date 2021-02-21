import string
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords


# Text data pre-processing
def text_processing(corpus):
    """
    A function for text processing. Including removing digits, punctuations, white space and stop words, transforming
    the input sentences into lower case, and finally tokenizing the sentences and stemming all the tokens.
    :param corpus: corpus to be processed
    :return: processed corpus
    """
    # build a function to remove things that are going to be removed
    def full_remove(x, removal_list):
        for w in removal_list:
            x = x.replace(w, '')
        return x

    # Remove digits from input text
    remove_digits = [full_remove(x, list(str(x) for x in range(10))) for x in corpus]

    # Remove punctuation from input text
    remove_punctuation = [full_remove(x, list(string.punctuation)) for x in remove_digits]

    # Remove any white space and transform all the inout text into lower case
    sentence_lower = [x.lower().strip() for x in remove_punctuation]

    # Tokenize the input sentence
    sentence_tokenized = [word_tokenize(x) for x in sentence_lower]

    # POS tagging
    sentence_pos_tagged = [nltk.pos_tag(x) for x in sentence_tokenized]

    # build a function to remove stop words and noun phrase
    def remove_stop_words_and_noun(stop_words, sentence):
        filtered_sentence = []
        for token in sentence:
            if token[0] not in stop_words and not token[1].startswith("NN"):
                filtered_sentence.append(token[0])
        return filtered_sentence

    # Remove stop words from input sentence
    stops = stopwords.words("English")
    # stop_set = ['the', 'a', 'an', 'i', 'he', 'she', 'they', 'to', 'of', 'it', 'from']
    sentence_filtered = [remove_stop_words_and_noun(stops, x) for x in sentence_pos_tagged]

    # Build a function to lemmatize tokens
    def lemmatize(words):
        wnl = nltk.WordNetLemmatizer()
        lemmatized_words = [wnl.lemmatize(t) for t in words]
        return lemmatized_words

    # Lemmatizing tokens
    tokens_lemmatized = [lemmatize(x) for x in sentence_filtered]
    sentence_lemmatized = [' '.join(x) for x in tokens_lemmatized]

    return sentence_lemmatized
