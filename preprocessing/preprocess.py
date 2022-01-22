import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

#This module handles all the preprocessing operations.

#Download requested modules from nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

#List of characters to be removed
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;:".!?\-]')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def strip_text(text):

    '''
    Strip the given text
    '''

    return text.strip()

def set_to_lower(text):

    '''
    Lower the given text
    '''

    return text.lower()

def replace_with_spaces(text):

    '''
    Remove unwanted characters
    '''

    return REPLACE_BY_SPACE_RE.sub(' ', text)

def get_words(text, remove_special_symbols=True):

    '''
    Split the text into words using the nltk toolkit
    '''

    words = word_tokenize(text)
    if remove_special_symbols:
        filtered_words = [word for word in words if word.isalnum()] #remove special symbols and punctuation
        return filtered_words
    else:
        return words

def get_wordnet_pos(word):

    '''
    Get the pos tag of a given word using the nltk toolkit
    '''

    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN) #Return the correspondent tag or NOUN if it is not in tag_dict

def lemmatize(words):

    '''
    Lemmatize a list of words using nltk's lemmatizer
    '''

    lemmatized_words = [lemmatizer.lemmatize(word,get_wordnet_pos(word)) for word in words]
    return lemmatized_words

def delete_stop_words(words):

    '''
    Remove stop words from a list of words
    '''

    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

def split_to_chars(word):

    '''
    Split a word into a list of characters
    '''

    return [char for char in word]

def preprocess_text(text, preprocessing_options, get_full_text=False):

    '''
    Apply the entire preprocess pipeline to a given text. The pipeline is defined by
    "preprocessing_options".
    '''

    if preprocessing_options["strip"]:
        text = strip_text(text)

    if preprocessing_options["lower"]:
        text = set_to_lower(text)

    if preprocessing_options["replace"]:
        text = replace_with_spaces(text)

    words = get_words(text, preprocessing_options["remove special"])
    full_text = words[0:] if get_full_text else None

    if preprocessing_options["stopwords"]:
        words = delete_stop_words(words)

    if preprocessing_options["lemmatize"]:
        words = lemmatize(words)

    if full_text is not None:
        return words, full_text

    else:
        return words
