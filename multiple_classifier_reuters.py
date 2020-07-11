#!/usr/bin/python3

from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers

from plot import Plot
from one_hot_encoding import one_hot_encoding
from evaluation import evaluate

# Configuration

# limit the training data to the max occurring number of words
_NUM_WORDS = 2000 # 10000
_EPOCHS_TRAIN = 2 # 20
_EPOCHS_EVAL = 5 # 7
_BATCH_SIZE = 512
_VERBOSE = 1

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(_NUM_WORDS,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(46, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# find the correct num of epochs before applying model to test_data
def train(train_data, train_labels):
    # From the 8982 samples we wanna use 1.000 as validation data
    num_validation = 1000
    validation_train_data = train_data[:num_validation]
    partial_train_data = train_data[num_validation:]

    validation_train_labels = train_labels[:num_validation]
    partial_train_labels = train_labels[num_validation:]

    model = build_model()

    history = model.fit(partial_train_data, partial_train_labels,
                        epochs=_EPOCHS_TRAIN, batch_size=_BATCH_SIZE, verbose=_VERBOSE,
                        validation_data=(validation_train_data, validation_train_labels))

    # plot history
    p = Plot(history, "Multiple-Classifier-Reuters", "multiplie-classifier-reuters")
    p.plot()


if __name__ == '__main__':
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=_NUM_WORDS)

    binary_train_data = one_hot_encoding(train_data, _NUM_WORDS)
    binary_test_data = one_hot_encoding(test_data, _NUM_WORDS)

    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)

    train(binary_train_data, one_hot_train_labels)
    evaluate(build_model(), binary_train_data, one_hot_train_labels, binary_test_data, one_hot_test_labels,
             _EPOCHS_EVAL, _BATCH_SIZE, _VERBOSE)
