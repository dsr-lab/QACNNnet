# This module handles all GloVe-related operations

import numpy as np
import os

BASE_PATH = os.path.join(os.getcwd(), "data","fastText")

def load_fast_text(language, vocabulary):
  '''
  Open fastText's txt file and store each of its contained words
  into a dictionary along with their correspondent embedding weights.

  Parameters:
  --------
  language: string
    Language embedding to be loaded

  Returns:
  -------
  vocabulary: Dict
      fastText's vocabulary
  '''

  print("Loading fastText Model...")

  path = os.path.join(BASE_PATH, "wiki.{}.align.vec".format(language))

  with open(path, encoding="utf8" ) as f: #Open the txt file
      lines = f.readlines() #Read the file line by line

  first_line = lines[0].split()
  print("There are", first_line[0], "elements in the vocabulary of the language", language)
  for line in lines[1:]:
      splits = line.split()
      # Save the first part of the line (word) as the dictionary's key and the second part (the embedding) as the key
      embedding = []
      for val in splits[1:]:
        try:
            float_value = float(val)
            embedding.append(float_value)
        except ValueError:
            print("Skipped invalid value:", val)

      key = (language, splits[0])
      vocabulary[key] = np.array(embedding)

  print("fastText model loaded!")

  return vocabulary
