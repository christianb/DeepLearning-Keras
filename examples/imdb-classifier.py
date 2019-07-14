#  IMDB  50.000 movie ratings.
# 25.000 train sequences and 25.000 test sequences.

from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers

_NUM_WORDS=5000

def run_validation_model(train_data, train_labels, test_data, test_labels):
	# Model creation
	model =  models.Sequential()
	model.add(layers.Dense(16, activation='relu',  input_shape=(_NUM_WORDS,)))
	model.add(layers.Dense(16, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))

	model.compile(optimizer='rmsprop',
			loss='binary_crossentropy',
			metrics=['accuracy'])

	# Validation
	# From the 25.000 samples we wanna use 10.000 as validation data
	NUM_VALIDATION = 10000
	validation_train_data = train_data[:NUM_VALIDATION]
	partial_train_data = train_data[NUM_VALIDATION:]

	validation_train_labels = train_labels[:NUM_VALIDATION]
	partial_train_labels = train_labels[NUM_VALIDATION:]

	# Run the training
	history = model.fit(partial_train_data,
		  	    partial_train_labels,
		  	    epochs=20,
		  	    batch_size=512,
		  	    validation_data=(validation_train_data, validation_train_labels))

	# Print the results
	import matplotlib.pyplot as plt

	# Values Loss Function
	history_dict = history.history
	loss_values = history_dict['loss']
	validation_loss_values = history_dict['val_loss']
	epochs = range(1, len(loss_values) + 1)

	plt.figure()
	plt.plot(epochs, loss_values, 'bo', label="Loss Training")
	plt.plot(epochs, validation_loss_values, 'b', label='Loss Validation')
	plt.title('Loss function results')
	plt.xlabel('Epochs')
	plt.ylabel('Value loss function')
	plt.legend()

	# Values Accuracy
	acc_values = history_dict['acc']  
	validation_acc_values = history_dict['val_acc']

	plt.figure()
	plt.plot(epochs, acc_values, 'bo', label='Accuracy Traning')
	plt.plot(epochs, validation_acc_values, 'b', label='Accuracy Validation')
	plt.title('Accuracy Results')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy value')
	plt.legend()

	plt.show()
	print("result: ", model.evaluate(test_data, test_labels))

def run_model(train_data, train_labels, test_data, test_labels):
	model = models.Sequential()
	model.add(layers.Dense(16, activation='relu', input_shape=(_NUM_WORDS,)))
	model.add(layers.Dense(16, activation='relu'))
	model.add(layers.Dense(1, activation='sigmoid'))
	
	model.compile(optimizer='rmsprop', loss='binary_crossentropy',  metrics=['accuracy'])

	model.fit(train_data, train_labels, epochs=4, batch_size=512)

	print("result: ", model.evaluate(test_data, test_labels))
	print("prediction for each test sample: ",model.predict(test_data))

def main():
	# using more words needs more RAM, or MemoryError is thrown
	# num_words is the number of the most occuring words in the ratings  

	(train_data, train_labels), (test_data, test_labels) =  imdb.load_data(num_words=_NUM_WORDS)

	print("len(train_data): ", len(train_data))
	print("len(test_data): ", len(test_data))

	# print("train_data[0]: ", train_data[0])
	# print("train_labels[0]: ", train_labels[0])

	# convert word index to words
	word_index = imdb.get_word_index()
	reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
	decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
	# print(decoded_review)

	# using larger dimensions needs more RAM, or MemoryError is thrown
	# creates a matrix with shape(len(sequences),  dimension).
	# One-Hot encoding
	def vectorize_sequences(sequences, dimension):
	  results = np.zeros(shape=(len(sequences), dimension), dtype=float)
	  for i, sequence in enumerate(sequences):
	    results[i, sequence] = 1.
	  return results

	binary_train_data = vectorize_sequences(train_data, _NUM_WORDS)
	binary_test_data = vectorize_sequences(test_data, _NUM_WORDS)
	# binary data contains a list of 0 and 1 only
	# print("binary_train_data[0]: ", binary_train_data[0])

	binary_train_labels = np.asarray(train_labels).astype(float)
	binary_test_labels = np.asarray(test_labels).astype(float)

	run_validation_model(binary_train_data, binary_train_labels, binary_test_data, binary_test_labels)
	# run_model(binary_train_data, binary_train_labels, binary_test_data, binary_test_labels)
    
if __name__ == '__main__':
	main()

