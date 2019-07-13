from keras.datasets import imdb

# using more words needs more RAM, or MemoryError is thrown
(train_data, train_labels), (test_data, test_labels) =  imdb.load_data(num_words=5000)

# print("train_data[0]: ", train_data[0])
# print("train_labels[0]: ", train_labels[0])

# convert word index to words
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
# print(decoded_review)

import numpy as np

# using larger dimensions needs more RAM, or MemoryError is thrown
def vectorize_sequences(sequences, dimension=5000):
  print("len(sequences): ", len(sequences))
  results = np.zeros(shape=(len(sequences), dimension), dtype=float)
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results

vectorized_train_data = vectorize_sequences(train_data)
vectorized_test_data = vectorize_sequences(test_data)
print("vectorized_train_data[0]: ", vectorized_train_data[0])

vectorized_train_labels = np.asarray(train_labels).astype(float)
vectorizes_test_labels = np.asarray(test_labels).astype(float)

