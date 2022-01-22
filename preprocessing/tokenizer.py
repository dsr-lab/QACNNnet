import numpy as np

#This module hanldes all the tokenization operations.

def get_unique_words(text_rows):

    '''
    Build set of unique words in a given text.
    '''

    unique_words = set()
    for row in text_rows:
        context_words = set(row["Context words"])
        question_words = set(row["Question words"])

        unique_words = unique_words | context_words | question_words

    return unique_words

def get_unique_chars(text_rows):

    '''
    Build set of unique characters in a given text.
    '''

    unique_chars = set()
    for row in text_rows:
        context_chars = row["Context chars"]
        question_chars = row["Question chars"]

        chars_lists = context_chars + question_chars

        for char_list in chars_lists:
            chars = set(char_list)
            unique_chars = unique_chars | chars

    return unique_chars

def build_words_tokenizer(unique_words, glove):

    '''
    Build tokenizer dictionary containing an integer value for each
    unique word that has an entry in GloVe's dictionary. All the other
    words ("UNK") are mapped into a last integer value.
    '''

    vocab = {}
    for word in unique_words:
        if word in glove:
            vocab[word]=len(vocab)+1

    vocab["UNK"] = len(vocab)+1

    return vocab

def build_chars_tokenizer(unique_chars):

    '''
    Build tokenizer dictionary containing an integer value for each
    unique character.
    '''

    vocab = {}
    for char in unique_chars:
        vocab[char]=len(vocab)+1

    vocab["unk"] = len(vocab)+1

    return vocab

def tokenize_word(word, tokenizer):

    '''
    Return the token of a given word inside the built tokenizer.
    '''

    if word in tokenizer:
        return tokenizer[word]
    else:
        return tokenizer["UNK"]

def tokenize_char(char,tokenizer):

    '''
    Return the token of a given character inside the built tokenizer.
    '''

    if char in tokenizer:
        return tokenizer[char]
    else:
        return tokenizer["unk"]

def tokenize_char_sequence(char_sequence, tokenizer):

    '''
    Return the tokenized sequence of a given list of characters.
    '''

    return [tokenize_char(char,tokenizer) for char in char_sequence]

def add_padding_or_truncate(tokenized_sequence, max_length, char_mode=False):

    '''
    Given a tokenized sequence and a maximum length, return the same sequence
    so that it is truncated or padded to reach exactly the maximum length.
    '''

    length_diff = max_length-len(tokenized_sequence)

    if length_diff==0: # Return the same sequence if it has the requested size
        return tokenized_sequence
    elif length_diff<0: # Return the truncated sequence if it exceeds the requested size
        return tokenized_sequence[0:max_length]
    else: # Return the padded sequence if it has an inferior size than the expected one
        if char_mode:
            return np.pad(tokenized_sequence, ((0, length_diff),(0,0)), 'constant')
        else:
            return np.pad(tokenized_sequence, (0, length_diff), 'constant')

def pad_truncate_tokenize_words(words, tokenizer, max_words):

    '''
    Tokenize and then apply padding or truncation to a list of words.
    '''

    tokenized_sequence = [tokenize_word(word, tokenizer) for word in words]
    return add_padding_or_truncate(tokenized_sequence, max_words)

def pad_truncate_tokenize_chars(chars, tokenizer, max_chars):

    '''
    Tokenize and then apply padding or truncation to a list of characters.
    '''

    tokenized_sequence = [tokenize_char(char, tokenizer) for char in chars]
    return add_padding_or_truncate(tokenized_sequence, max_chars)

def pad_truncate_tokenize_chars_sequence(chars_lists, tokenizer, max_words, max_chars):

    '''
    Tokenize and then apply padding or truncation to a list of lists of characters.
    '''

    tokenized_sequence = [pad_truncate_tokenize_chars(char_list, tokenizer, max_chars) for char_list in chars_lists]
    return add_padding_or_truncate(tokenized_sequence, max_words, char_mode=True)

def detokenize(token, tokenizer):

    '''
    Return a word or character, given its token value and the correspondent
    tokenizer.
    '''

    val_list = list(tokenizer.values())
    key_list = list(tokenizer.keys())

    position = val_list.index(token)

    return key_list[position]
