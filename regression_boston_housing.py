#!/usr/bin/python3

from keras import models
from keras import layers
from keras.datasets import boston_housing

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Configuration

_EPOCHS_TRAIN = 1 # 200
_BATCH_SIZE = 16
_VERBOSE = 1

def build_model(train_data):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def train_and_evaluate_with_k_cross_validation(train_data, train_targets):
    k = 4

    num_val_samples = len(train_data) // k

    all_scores = []
    all_mae_histories = []

    for i in range(k):
        # print('Run #', i)
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                            axis=0)
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

        model = build_model(train_data)

        # train
        history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                            epochs=_EPOCHS_TRAIN, batch_size=_BATCH_SIZE, verbose=_VERBOSE)

        # evaluate
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=_VERBOSE)

        print(history.history.keys())

        mae_history = history.history['mae']

        all_scores.append(val_mae)
        all_mae_histories.append(mae_history)

    print('scores from cross-validation: ', all_scores)
    print('mean all scores: ', np.mean(all_scores))

    # return average_mae_history
    return [np.mean([x[i] for x in all_mae_histories]) for i in range(_EPOCHS_TRAIN)]


def evaluate(train_data, train_targets, test_data, test_targets):
    model = build_model(train_data)
    model.fit(train_data, train_targets, epochs=_EPOCHS_TRAIN, batch_size=_BATCH_SIZE, verbose=_VERBOSE)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)

    return smoothed_points


if __name__ == '__main__':
    (train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

    # normalize data
    mean = train_data.mean(axis=0)
    train_data -= mean
    std = train_data.std(axis=0)
    train_data /= std

    test_data -= mean  # Always use values calculated from the train data, never from the test data!
    test_data /= std

    average_mae_history = train_and_evaluate_with_k_cross_validation(train_data, train_targets)

    smooth_mae_history = smooth_curve(average_mae_history[10:])

    plt.clf()
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Mean absolut error - Validation')
    plt.savefig('regression-loss.png')
    # plt.legend()
