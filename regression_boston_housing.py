from keras import models
from keras import layers
from keras.datasets import boston_housing

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Configuration

_EPOCHS_TRAIN = 10
_BATCH_SIZE = 1
_VERBOSE = 1


def build_model(train_data):
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# k_cross_validation
def train_and_evaluate(train_data, train_targets):
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

        history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                            epochs=_EPOCHS_TRAIN, batch_size=_BATCH_SIZE, verbose=_VERBOSE)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=_VERBOSE)

        print(history.history.keys())

        mae_history = history.history['mae']

        all_scores.append(val_mae)
        all_mae_histories.append(mae_history)

    print('scores from cross-validation: ', all_scores)
    print('mean all scores: ', np.mean(all_scores))

    # return average_mae_history
    return [np.mean([x[i] for x in all_mae_histories]) for i in range(_EPOCHS_TRAIN)]


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# normalize data
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean  # TODO could we calculate the mean and std from test_data itself?
test_data /= std

average_mae_history = train_and_evaluate(train_data, train_targets)

plt.clf()
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Mean absolut error - Validation')
plt.legend()
plt.savefig('regression.png')
