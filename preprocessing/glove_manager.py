import os
import urllib
import zipfile
import numpy as np
import Config

#This module handles all GloVe-related operations.

URL_BASE = "https://nlp.stanford.edu/data" #Location of the pre-trained GloVe's files
GLOVE_VERSION = "6B"

EMBEDDING_SIZE = Config.WORD_EMBEDDING_SIZE

#List of paths to download and extract GloVe's files
PATHS = {
    "url": URL_BASE + "/glove." + GLOVE_VERSION + ".zip",
    "glove_path": os.path.join(os.getcwd(), "data","Glove",GLOVE_VERSION),
    "glove_zip": os.path.join(os.getcwd(), "data","Glove", GLOVE_VERSION, "glove."+GLOVE_VERSION+".zip"),
    "glove_file": os.path.join(os.getcwd(), "data","Glove", GLOVE_VERSION, "glove."+GLOVE_VERSION+"."+str(EMBEDDING_SIZE)+"d.txt")
}

def setup_files():
  '''
  Create the folder if it does not exist.
  Then download the zip file from the web archive if it does not exist.
  Finally extract the zip file of the GloVe txt file does not exist in the folder.
  '''

  if not os.path.exists(PATHS["glove_path"]):
    os.makedirs(PATHS["glove_path"])

  if not os.path.exists(PATHS["glove_file"]):
    if not os.path.exists(PATHS["glove_zip"]):
      download_glove(PATHS["url"])

    extract_glove(PATHS["glove_zip"],PATHS["glove_path"])

def download_glove(url):
    '''
    Download GloVe's zip file from the web.
    '''

    print("Downloading GloVe...")
    urllib.request.urlretrieve(url, PATHS['glove_zip'])
    print("Successful download")

def extract_glove(zip_file, glove_path):

    '''
    Extract GloVe's zip file.
    '''

    print("Extracting GloVe...")
    with zipfile.ZipFile(PATHS["glove_zip"], 'r') as zip_ref:
      zip_ref.extractall(path=PATHS["glove_path"])
      print("Successful extraction!")

def load_glove():
  '''
  Open GloVe's txt file and store each of its contained words
  into a dictionary along with their correspondent embedding weights.

  Returns:
  -------
  vocabulary: Dict
      GloVe's vocabulary

  '''
  print("Loading GloVe Model...")

  with open(PATHS["glove_file"], encoding="utf8" ) as f: #Open the txt file
      lines = f.readlines() #Read the file line by line

  vocabulary = {}
  for line in lines:
      splits = line.split()
      #Save the first part of the line (word) as the dictionary's key and the second part (the embedding) as the key
      vocabulary[splits[0]] = np.array([float(val) for val in splits[1:]])

  print("GloVe model loaded!")

  return vocabulary
