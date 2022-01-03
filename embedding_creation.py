import os
import re
import requests
import zipfile
import numpy as np
import pandas as pd
from functools import reduce, partial
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize,WhitespaceTokenizer

# typing
from typing import List, Callable, Dict
DATASET_NAME = "training_set" #Name of the folder that will be automatically created after extracting the dataset

#List of paths to handle the dataset
DATASET_PATHS = {
    "dataset_folder": os.path.join(os.getcwd(), "Datasets"), # Folder containing the original dataset data
    "dataset_path" : os.path.join(os.getcwd(), "Datasets", "training_set.json"), # Path to zipped dataset
    "documents_path" : os.path.join(os.getcwd(), "Datasets", DATASET_NAME), # Folder containing extracted documents (NB: it is created automatically during the extraction)
    "dataframe_folder" : os.path.join(os.getcwd(), "Datasets", "Dataframes", DATASET_NAME), # Folder containing the dataframe data
    "dataframetrain_path" : os.path.join(os.getcwd(), "Datasets", "Dataframes", DATASET_NAME, DATASET_NAME + "_train.pkl"),
    "dataframeval_path" : os.path.join(os.getcwd(), "Datasets", "Dataframes", DATASET_NAME, DATASET_NAME + "_val.pkl") # Path to pickle save of built dataframe
}

def create_folders(paths: List[str]):
  '''
  Create the folders in paths list.

  Parameters:
  ----------
  paths: List[str]
    A list of all the folders to create
  '''

  for path in paths:
    if not os.path.exists(path):
      os.makedirs(path)

folders = [DATASET_PATHS["dataframe_folder"]]
           
create_folders(folders)

def build_dataframe(dataframetrain_path: str, dataframeval_path: str, save = True)->(pd.DataFrame, pd.DataFrame):
# load data using Python JSON module
	with open(DATASET_PATHS["dataset_path"],'r') as f:
		data = json.loads(f.read())

	data_train = data["data"][:376]
	data_val = data["data"][377:]
	df_train = pd.json_normalize(data_train, record_path =['paragraphs', 'qas', 'answers'], meta=[['title'], ['paragraphs', 'context'], ['paragraphs', 'qas', 'question']])
	df_val = pd.json_normalize(data_val, record_path =['paragraphs', 'qas', 'answers'], meta=[['title'], ['paragraphs', 'context'], ['paragraphs', 'qas', 'question']])
	print("Dataframe built successfully")
	if save:
		df_train.to_pickle(dataframetrain_path)
		df_val.to_pickle(dataframeval_path)
		print("Dataframe saved successfully")
 
	return df_train, df_val

def load_dataframe(dataframetrain_path: str, dataframval_path: str, force_rebuild = False)->pd.DataFrame:
	'''
	Load the dataframe from memory if it is possible, or build a new one.

	Parameters:
	----------
	documents_path: str
		Path containing the documents
    dataframe_path: str
		Path to load the dataframe from
	force_rebuild: bool
		Flag to force the rebuild of the dataframe even if a saved copy exists

	Returns:
	--------
	pd.DataFrame
		The built or loaded dataframe
	'''
	if not os.path.exists(dataframetrain_path) or not os.path.exists(dataframeval_path) or force_rebuild:
		return build_dataframe(dataframetrain_path, dataframeval_path)
	else:
		df_train = pd.read_pickle(dataframetrain_path)
		df_val = pd.read_pickle(dataframeval_path)
		return df_train, df_val #Load dataframe

#df.rename(columns={"data.title ": "title", "data.paragraphs.context ": "context", "data.paragraphs.qas.question": "question"})
df_train, df_val = build_dataframe(DATASET_PATHS["dataframetrain_path"], DATASET_PATHS["dataframeval_path"])

# Special characters to remove: /(){}[]|@,;
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')

# Accepted symbols:
# - numbers between 0-9
# - all lower cased letters
# - whitespace, #, + _
GOOD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

# - ^ begnning of a string
# - \d any digit
# - \s whitespaces and tabs
BEGINNING_IDS_RE = re.compile('^\d*\s*')

# Remove multiple whitespaces, tabs and newlines
EXTRA_WHITE_SPACE_RE = re.compile('/\s\s+/g')

# The stopwords are a list of words that are very very common but don’t 
# provide useful information for most text analysis procedures.
# Therefore, they will be removed from the dataset
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    STOPWORDS = set(stopwords.words('english'))


nltk.download('punkt') # necessary for being able to tokenize
nltk.download('wordnet') 
nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()
tokenizer = WhitespaceTokenizer()

from nltk.corpus import wordnet
def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def replace_special_characters(text: str) -> str:
    """
    Replaces special characters, such as paranthesis,
    with spacing character
    """

    return REPLACE_BY_SPACE_RE.sub(' ', text)

def lower(text: str) -> str:
    """
    Transforms given text to lower case.
    Example:
    Input: 'I really like New York city'
    Output: 'i really like new your city'
    """

    return text.lower()

def filter_out_uncommon_symbols(text: str) -> str:
    """
    Removes any special character that is not in the
    good symbols list (check regular expression)
    """

    return GOOD_SYMBOLS_RE.sub('', text)

def remove_stopwords(text: str) -> str:
    """
    Method used for removing most common words

    Parameters
    ----------
    text : str
        The text to process
    
    Returns
    -------
    text : str
        The processed text.
    """
    return ' '.join([x for x in text.split() if x and x not in STOPWORDS])

def strip_text(text: str) -> str:
    """
    Removes any left or right spacing (including carriage return) from text.
    Example:
    Input: '  This assignment is cool\n'
    Output: 'This assignment is cool'
    """

    return text.strip()

def replace_ids(text: str) -> str:
    """
    Method used for removing ids and some whitespaces that could appear
    at the beginning of the text.

    Parameters
    ----------
    text : str
        The text to process
    
    Returns
    -------
    text : str
        The processed text.
    """
    return BEGINNING_IDS_RE.sub('', text)

def lemsent(sentence):
    """
    Method used for lemmatize text.

    Parameters
    ----------
    text : str
        The text to process.
    
    Returns
    -------
    text : str
        The processed text.
    """
    #words = [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(str(sentence))]
    words = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)]
    return " ".join(words)

#List of all the preprocessing methods to be called
GENERIC_PREPROCESSING_PIPELINE = [
                                  lower,
                                  replace_special_characters,
                                  filter_out_uncommon_symbols,
                                  #remove_stopwords,
                                  strip_text,
                                  lemsent
                                  ]

def text_prepare(text: str,
                 filter_methods: List[Callable[[str], str]] = None) -> str:
    """
    Applies a list of pre-processing functions in sequence (reduce).
    Note that the order is important here!
    """
    filter_methods = GENERIC_PREPROCESSING_PIPELINE

    return reduce(lambda txt, f: f(txt), filter_methods, text)

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
  '''
  Apply the preprocessing operations to the dataframe.

  Parameters
  ----------
  df: pd.DataFrame
    Dataframe to be preprocessed

  Returns
  --------
  df: pd.DataFrame
    Preprocessed dataframe
  '''

  # Replace each sentence with its pre-processed version
  df['text'] = df['text'].apply(lambda txt: text_prepare(txt))
  df['title'] = df['data.title'].apply(lambda txt: text_prepare(txt))
  df['paragraphs.context'] = df['data.paragraphs.context'].apply(lambda txt: text_prepare(txt))
  df['paragraphs.qas.question'] = df['data.paragraphs.qas.question'].apply(lambda txt: text_prepare(txt))
  
  return df

#df_train = preprocess_dataset(df_train)
#df_val = preprocess_dataset(df_val)
print(df_train.loc[0])
print(df_train.info())
print(df_val.loc[0])
print(df_val.info())

STARTING_TOKEN = 1 #First value to start the tokenization on (0 is already used as padding value)

def get_tokenizer(corpus: List[str],
                  starting_dict=None)->Dict[str,int]:
  '''
  Create or expand (given an existing dictionary) a tokenization dictionary
  that associates an integer to each word.

  Parameters:
  -----------
  corpus: List[str]
    Text to examine searching for new words to add into the dictionary
  starting_dict: Dict[str,int]
    An already filled dictionary to further expand (optional)

  Returns:
  --------
  words_to_tokens: Dict[str,int]
    1. A filled dictionary that associates an integer to each word (if starting_dict=None);
    2. An expanded dictionary that associates an integer to each new word (if starting_dict is not None)
  '''

  #Copy the original dictionary to keep it save from updates
  words_to_tokens = {} if starting_dict==None else starting_dict.copy()

  for text in corpus:
    words = text.split()
    for word in words:
      if not word in words_to_tokens:
        words_to_tokens[word] = len(words_to_tokens)+STARTING_TOKEN

  return words_to_tokens

def tokenize(word: str,
             words_to_tokens: Dict[str,int])->int:
  '''
  Get the integer value of a given token.

  Parameters:
  -----------
  word: str
    Token
  words_to_tokens: Dict[str,int]
    Tokenization dictionary

  Returns:
  -------
  int:
    Value associated to the token
  '''
  return words_to_tokens[word]

def detokenize(token:int,
               words_to_tokens: Dict[str,int])->str:
  '''
  Get the token-word of a given token-value.

  Parameters:
  -----------
  token: int
    Tokenized word
  words_to_tokens: Dict[str,int]
    Tokenization dictionary

  Returns:
  -------
  str:
    Word associated to the token-value
  '''
  val_list = list(words_to_tokens.values())
  key_list = list(words_to_tokens.keys())

  position = val_list.index(token)

  return key_list[position]

  #return words_to_tokens.index(token)

def tokenize_string(string: str,
                    words_to_tokens: Dict[str,int],
                    max_length: int)->List[int]:

  '''
  Get the tokenized sequence of a string of separated tokens (document/sentence).

  Parameters:
  string: str
    String of separated tokens (document or sentence)
  words_to_tokens: Dict[str,int]
    Tokenization dictionary
  max_length: int
    Tokenization length

  Returns:
    List[int]:
      A list of token-values where each one is the tokenized value of a token
      int the input-string.
      The list is padded if its length is below the max_length.
      The list is truncated if its length is above the max_length.
  '''
  tokens = string.split()
  tokenized_sequence = [tokenize(token, words_to_tokens)  for token in tokens]
  length_diff = max_length-len(tokenized_sequence)

  if length_diff==0: # Return the same sequence if it has the requested size
    return tokenized_sequence
  elif length_diff<0: # Return the truncated sequence if it exceeds the requested size
    return tokenized_sequence[0:max_length]
  else: # Return the padded sequence if it has an inferior size than the expected one
    return np.pad(tokenized_sequence, (0, length_diff), 'constant').tolist()

def get_list_of_answers(data: pd.DataFrame)->List[List[int]]:
  '''
  Returns a list of tuples with the beginning and the ending ids of the answers in the contexts.
  ------
  Parameters:
  data: pd.DataFrame
      The dataframe with the answers.
  ------
  Returns:
      answers_tuples: List[List[int]]
      The list of tuples of type (answer_start, answer_end)
  '''
	answers_start = data["answer_start"].tolist()
	answers_text = data["text"].tolist()
	answers_tuples = []
	for (idx, answer_start) in enumerate(answers_start):
		answer_tuple = []
		answer_tuple.append(answer_start)
    answers_text_list = answers_text[idx].split()
    answers_text_list = answers_text_list[:30]
    #answer_text = " ".join(answers_text_list)
		answer_tuple.append(answer_start + len(answers_text_list))
		answers_tuples.append(answer_tuple)
	return answers_tuples

#Define corpus
context_train = df_train["paragraphs.context"].tolist()
context_val = df_val["paragraphs.context"].tolist()
question_train = df_train["paragraphs.qas.question"].tolist()
question_val = df_val["paragraphs.qas.question"].tolist()
answer_train = get_list_of_answers(df_train)
answer_val = get_list_of_answers(df_val)


#Token dictionaries
context_train_tokens = get_tokenizer(context_train)
question_train_tokens = get_tokenizer(question_train, starting_dict = context_train_tokens)
context_val_tokens = get_tokenizer(context_val, starting_dict = question_train_tokens)
question_val_tokens = get_tokenizer(question_val, starting_dict = context_val_tokens)

#Vocabularies
vocabulary = question_val_tokens.keys()

#Vocab sizes
vocab_size = len(vocabulary)


#Tokenized sets
context_train_tokenized = np.array(list(map(lambda string: tokenize_string(string, context_train_tokens,400),context_train)))
context_val_tokenized = np.array(list(map(lambda string: tokenize_string(string, context_val_tokens,400),context_val)))
question_train_tokenized = np.array(list(map(lambda string: tokenize_string(string, question_train_tokens,50),question_train)))
question_val_tokenized = np.array(list(map(lambda string: tokenize_string(string, question_val_tokens,50),question_val)))
print(context_train_tokenized)

URL_BASE = "https://nlp.stanford.edu/data" #Location of the pre-trained GloVe's files
GLOVE_VERSION = "6B"

EMBEDDING_SIZE = 300 #The dimensionality of the embeddings; to be tested

#List of paths to download and extract GloVe's files
PATHS = {
    "url": URL_BASE + "/glove." + GLOVE_VERSION + ".zip",
    "glove_path": os.path.join(os.getcwd(),"Glove",GLOVE_VERSION),
    "glove_zip": os.path.join(os.getcwd(),"Glove", GLOVE_VERSION, "glove."+GLOVE_VERSION+".zip"),
    "glove_file": os.path.join(os.getcwd(),"Glove", GLOVE_VERSION, "glove."+GLOVE_VERSION+"."+str(EMBEDDING_SIZE)+"d.txt")
}

# Constant value used when OOV_METHOD = 'Placeholder'. Randomly initialized.
PLACEHOLDER = np.random.uniform(low=-0.05, high=0.05, size=EMBEDDING_SIZE)

def setup_files():
  '''
  Create the folder if it does not exist.
  Then download the zip file from the web archive if it does not exist.
  Finally exctract the zip file of the GloVe txt file does not exist in the folder.
  '''

  if not os.path.exists(PATHS["glove_path"]):
    os.makedirs(PATHS["glove_path"])

  if not os.path.exists(PATHS["glove_file"]):
    if not os.path.exists(PATHS["glove_zip"]):
      download_glove(PATHS["url"])

    extract_glove(PATHS["glove_zip"],PATHS["glove_path"])

def download_glove(url: str):
    '''
    Download GloVe's zip file from the web.
    '''

    urllib.request.urlretrieve(url, PATHS['glove_zip'])
    print("Successful download")

def extract_glove(zip_file: str,
                  glove_path: str):
  
    '''
    Extract GloVe's zip file.
    '''
  
    with zipfile.ZipFile(PATHS["glove_zip"], 'r') as zip_ref:
      zip_ref.extractall(path=PATHS["glove_path"])
      print("Successful extraction")

def load_model(glove_file: str) ->Dict:
  '''
  Open GloVe's txt file and store each of its contained words
  into a dictionary along with their correspondent embedding weights.

  Parameters:
  ----------
  glove_file : str
      GloVe's txt file path.

  Returns:
  -------
  vocabulary: Dict
      GloVe's vocabulary

  '''
  print("Loading GloVe Model...")

  with open(glove_file, encoding="utf8" ) as f: #Open the txt file
      lines = f.readlines() #Read the file line by line

  vocabulary = {}
  for line in lines:
      splits = line.split()
      #Save the first part of the line (word) as the dictionary's key and the second part (the embedding) as the key
      vocabulary[splits[0]] = np.array([float(val) for val in splits[1:]])

  print("GloVe model loaded")

  return vocabulary

def get_oov_list(words: List[str],
                 glove_embedding: Dict[str, int]) ->List[str]:
    '''
    Return a list of all the words that are not part of the GloVe embedding

    Parameters:
    ----------
    words: List[str]
        A list of unique words from a set of documents.
    glove_embedding: Dict[str, int]
        GloVe's embedding.

    Returns:
    -------
    oov: List[str]
        A list of all the OOV terms.
    '''
    embedding_vocabulary = set(glove_embedding.keys())
    oov = set(words).difference(embedding_vocabulary)

    return list(oov)

setup_files() #Create a path, download and extract the files, if necessary
glove_embedding = load_model(PATHS["glove_file"]) #Load the GloVe model

def update_embeddings(glove_embedding: Dict[str, int],
                     new_embeddings: Dict[str, int]):
    '''
    Update the GloVe's embeddings by adding the new embeddings of
    the previous OOV words.

    Parameters:
    ----------
    glove_embedding: Dict[str, int]
        GloVe's embedding.
    new_embeddings: Dict[str, int]
        A dictionary containing the new embeddings
        for the analyzed OOV words.
    '''
    
    #Merge GloVe's embeddings with the new discoveries
    glove_embedding.update(new_embeddings)

def build_embedding_matrix(vocab_size: int,
                            glove_embedding: Dict[str, int],
                            embedding_size: int,
                            words_to_tokens: Dict[str,int]) ->np.ndarray:
    '''
    Get the embedding matrix of the given set of documents/sentences.

    Parameters:
    -----------
    vocab_size: int
        Size of the set's vocabulary
    glove_embedding: Dict[str, int]
        GloVe's embedding
    embedding_size: int
        The embedding size for tokens' embeddings
    words_to_tokens: Dict[str,int]
        Tokenization dictionary of the given set

    Returns:
    --------
    embedding_matrix: np.ndarray
        Created and filled embedding matrix for the given set of documents/sentences
    '''
    embedding_matrix = np.zeros((vocab_size, embedding_size), dtype=np.float32) #Create an empty embedding matrix

    oov_terms = get_oov_list(words_to_tokens.keys(), glove_embedding)

    for word, token in tqdm(words_to_tokens.items()):
        if np.all((embedding_matrix[token-STARTING_TOKEN] == 0)):

            if word not in oov_terms: #Hanlde the OOV case with one of the methods
                embedding_vector = glove_embedding[word]

            embedding_matrix[token-STARTING_TOKEN] = embedding_vector #Update the embedding matrix

    #The computed values for the OOV words update the GloVe embeddings at the end of the process.
    #Updating these values at runtime affects the "Mean" OOV method.
    #update_embeddings(glove_embedding, discovered_embeddings)

    return embedding_matrix

#Build the embedding matrix
embedding_matrix = build_embedding_matrix(vocab_size,
                                                glove_embedding,
                                                EMBEDDING_SIZE,
                                                question_val_tokens)
def add_unk_embedding(embedding_matrix: np.ndarray)-> np.ndarray
  '''
  ------
  Add UNK embedding to the embedding matrix
  ------
  Parameters:
  embedding_matrix: np.ndarray 
      The embedding matrix.
  ------
  Returns:
  embedding_matrix: np.ndarray
      The embedding matrix with the UNK embedding vector added.
  '''
  unk_embedding = np.random.rand(1, 300)
  return embedding_matrix.appen(unk_embedding)

embedding_matrix = add_unk_embedding(embedding_matrix)
