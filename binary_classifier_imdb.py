#!/usr/bin/python3

import numpy as np

from keras import models
from keras import layers
from keras.datasets import imdb

from plot import Plot
from one_hot_encoding import one_hot_encoding
from evaluation import evaluate

# Configuration

# limit the training data to the max occurring number of words
_NUM_WORDS = 2000 # 10000
_EPOCHS_TRAIN = 5 # 20
_EPOCHS_EVAL = 4
_BATCH_SIZE = 512
_VERBOSE = 1

# help methods

def reverse(data):
    return dict([(value, key) for (key, value) in data.items()])


def decode_review(word_index, data, index):
    return ' '.join([word_index.get(i - 3, '?') for i in data[index]])


def get_review(imdb, data, index):
    reverse_word_index = reverse(imdb.get_word_index())
    return decode_review(reverse_word_index, data, index)


# model, building, evaluation and training

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(_NUM_WORDS,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# trains the model and validate the results
def train(train_data, train_labels, epochs):
    model = build_model()

    # Validation
    # From the 25.000 samples we wanna use 10.000 as validation data
    num_validation = 10000
    validation_train_data = train_data[:num_validation]
    partial_train_data = train_data[num_validation:]

    validation_train_labels = train_labels[:num_validation]
    partial_train_labels = train_labels[num_validation:]

    # Run the training
    history = model.fit(partial_train_data, partial_train_labels,
                        epochs=epochs,
                        batch_size=_BATCH_SIZE,
                        validation_data=(validation_train_data, validation_train_labels),
                        verbose=_VERBOSE)

    # plot history
    p = Plot(history, "Binary-Classifier-IMDB", "binary-classifier-imdb")
    p.plot()


if __name__ == '__main__':
    print("Load IMDB data. Select most " + str(_NUM_WORDS) + " words.")
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=_NUM_WORDS)

    # print(get_review(imdb, train_data, 0))

    print("prepare data and labels")
    binary_train_data = one_hot_encoding(train_data, _NUM_WORDS)
    binary_test_data = one_hot_encoding(test_data, _NUM_WORDS)

    binary_train_labels = np.asarray(train_labels).astype(float)
    binary_test_labels = np.asarray(test_labels).astype(float)

    train(binary_train_data, binary_train_labels, _EPOCHS_TRAIN)

    evaluate(build_model(),
             binary_train_data, binary_train_labels, binary_test_data, binary_test_labels,
             _EPOCHS_EVAL, _BATCH_SIZE, _VERBOSE)
