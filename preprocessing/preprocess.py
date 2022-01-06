import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

nltk.download('punkt')

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def strip_text(text):

    return text.strip()

def set_to_lower(text):

    return text.lower()

def replace_with_spaces(text):

    return REPLACE_BY_SPACE_RE.sub(' ', text)

def get_words(text, remove_special_symbols=True):

    words = word_tokenize(text)
    if remove_punctuation:
        filtered_words = [word for word in words if word.isalnum()] #remove special symbols and punctuation
        return filtered_words
    else:
        return words

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(words):

    lemmatized_words = [lemmatizer.lemmatize(word,get_wordnet_pos(word)) for word in words]
    return lemmatized_words

def delete_stop_words(words):

    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

def split_to_chars(text):

    return [char for char in word]

def preprocess_text(text, preprocessing_options):

    if preprocessing_options["strip"]:
        text = strip_text()

    if preprocessing_options["lower"]:
        text = set_to_lower(text)

    if preprocessing_options["replace"]:
        text = replace_with_spaces(text)

    words = get_words(text, preprocessing_options["remove special"])

    if preprocessing_options["stopwords"]:
        words = delete_stop_words(words)

    if preprocessing_options["lemmatize"]:
        words = lemmatize(words)

    return words
