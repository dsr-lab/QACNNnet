import tensorflow as tf
import numpy as np
import Config

#This module deals with computing the correct prediction from model's output.

inference_mask = None #This mask is computed just once and saved

def create_inference_mask(max_words):

    '''
    Create a mask used to make null those intervals where end>start.
    '''

    global inference_mask

    #Initialize mask
    mask = np.zeros((max_words,max_words), dtype=np.float32)

    #Fill mask with booleans values based on start<=end condition
    for start in range(max_words):
        for end in range(max_words):
            mask[start][end]=start<=end

    inference_mask = mask

def get_batched_inference_mask(batch_size, max_words):

    '''
    Get a mask that is repeated along the batch size.
    '''

    reshaped_inference_mask = tf.reshape(inference_mask,[1,max_words,max_words])
    batched_inference_mask = tf.repeat(reshaped_inference_mask,batch_size,axis=0)

    return batched_inference_mask

def get_predictions(predictions_start, predictions_end, n_words=Config.MAX_CONTEXT_WORDS):

    '''
    Compute the real predictions from model's output.
    For each batch:
    1. Create a matrix nxn for start's predictions (n=max number of tokens)
    2. Create a matrix nxn for end's prediction
    3. Compute thei product
    4. Get the max value from it
    5. Extract the indices of the max value and return them
    '''

    batch_size = tf.shape(predictions_start)[0]
    #n_words = predictions_start.shape[1]
    #n_words = Config.MAX_CONTEXT_WORDS

    if inference_mask is None:
        create_inference_mask(n_words)

    batched_inference_mask = get_batched_inference_mask(batch_size, n_words)

    #Reshape (batch_size, n_words)
    predictions_start = tf.reshape(predictions_start, [batch_size,n_words])
    predictions_end = tf.reshape(predictions_end, [batch_size,n_words])

    #Build product matrix (batch_size, n_words, n_words), first dim start and second dim end
    a = tf.repeat(predictions_start, n_words,axis=-1)
    a = tf.reshape(a, [batch_size,n_words,n_words])
    b = tf.repeat(predictions_end, n_words,axis=0)
    b = tf.reshape(b, [batch_size,n_words,n_words])
    predictions = tf.math.multiply(a, b)

    #Apply inference mask
    masked_predictions = tf.multiply(predictions, batched_inference_mask)

    #Flatten predictions and get argmax
    masked_predictions = tf.reshape(masked_predictions, [batch_size,n_words*n_words])
    argmax = tf.argmax(masked_predictions, axis=-1)

    #Get max's indices
    argmax_start = argmax // n_words
    argmax_start = tf.reshape(argmax_start, [batch_size,1])
    argmax_end = argmax % n_words
    argmax_end = tf.reshape(argmax_end, [batch_size,1])

    return argmax_start, argmax_end
