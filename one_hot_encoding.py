import numpy as np


# We can not give a (random length) list of integers to the neural network.
# So we have to transform them into Tensors of same length.
# If we have 10.000 words, a list of [5, 13, 42] (pointing to the words at these index)
# would transform into a vector with size of 10.000 which contains only Zeros except at the index of 5, 13 and 42 where it contains 1.
# This is called a One-hot encoding.
# Returns a matrix where each row is a one-hot encoded vector with a 1 at the index of the word
def one_hot_encoding(sequences, dimension):
    results = np.zeros(shape=(len(sequences), dimension), dtype=float)
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


if __name__ == '__main__':
    list = [3, 5]
    result = one_hot_encoding(list, 10)
    print(result)
