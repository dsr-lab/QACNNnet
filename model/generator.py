'''
This class creates batches to train the network on
'''

import config
import numpy as np
import tensorflow as tf
import math

#TODO: fix Eager mode necessary

class Generator(tf.keras.utils.Sequence):

    def __init__(self, input_train, output_train, input_validation,
                output_validation, shuffle=True):

        self.train_w_context, self.train_c_context, self.train_w_query, self.train_c_query = input_train
        self.valid_w_context, self.valid_c_context, self.valid_w_query, self.valid_c_query = input_validation

        self.output_train = output_train
        self.output_validation = output_validation

        # Check if it necessary to create a dummy dataset
        if config.DEBUG:
            self.create_dummy_dataset()

        # Check if it is necessary to remove the validation set
        if config.TRAIN_ON_FULL_DATASET:
            self.create_full_dataset()

        # Adjust tensor dimension
        self.output_train = np.expand_dims(self.output_train, -1)
        self.output_validation = np.expand_dims(self.output_validation, -1)

        if shuffle:
            self.shuffle_dataset()

    def create_dummy_dataset(self):

        n_train = 50
        n_val = 10

        self.train_w_context = self.train_w_context[:n_train]
        self.train_c_context = self.train_c_context[:n_train]
        self.train_w_query = self.train_w_query[:n_train]
        self.train_c_query = self.train_c_query[:n_train]
        self.output_train = self.output_train[:n_train]

        self.valid_w_context = self.valid_w_context[:n_val]
        self.valid_c_context = self.valid_c_context[:n_val]
        self.valid_w_query = self.valid_w_query[:n_val]
        self.valid_c_query = self.valid_c_query[:n_val]
        self.output_validation = self.output_validation[:n_val]

    def create_full_dataset(self):

        self.train_w_context = np.concatenate((self.train_w_context, self.valid_w_context), axis=0)
        self.train_c_context = np.concatenate((self.train_c_context, self.valid_c_context), axis=0)
        self.train_w_query = np.concatenate((self.train_w_query, self.valid_w_query), axis=0)
        self.train_c_query = np.concatenate((self.train_c_query, self.valid_c_query), axis=0)
        self.output_train = np.concatenate((self.output_train, self.output_validation), axis=0)

    def shuffle_dataset(self):

        permutation = np.random.permutation(len(self.train_w_context))

        self.train_w_context = self.train_w_context[permutation]
        self.train_c_context = self.train_c_context[permutation]
        self.train_w_query = self.train_w_query[permutation]
        self.train_c_query = self.train_c_query[permutation]

        self.output_train = self.output_train[permutation]

    def on_epoch_end(self):
        pass

    def __getitem__(self, idx):

        #Return next batch for training

        train_w_context = self.train_w_context[idx * config.BATCH_SIZE:(idx + 1) * config.BATCH_SIZE]
        train_c_context = self.train_c_context[idx * config.BATCH_SIZE:(idx + 1) * config.BATCH_SIZE]
        train_w_query = self.train_w_query[idx * config.BATCH_SIZE:(idx + 1) * config.BATCH_SIZE]
        train_c_query = self.train_c_query[idx * config.BATCH_SIZE:(idx + 1) * config.BATCH_SIZE]

        x = [train_w_context, train_c_context, train_w_query, train_c_query]
        y = self.output_train[idx * config.BATCH_SIZE:(idx + 1) * config.BATCH_SIZE]

        return x, y

    def __len__(self):

        return math.ceil(len(self.train_w_context) / config.BATCH_SIZE)
