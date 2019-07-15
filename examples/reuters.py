# Reuters 
# 8982 trainings samples
# 2246 test samples

from keras.datasets import reuters
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
 
_NUM_WORDS = 5000

def run_validation_model(train_data, train_labels, test_data, test_labels, model):
	# Validation
	# From the 8982 samples we wanna use 1.000 as validation data
	NUM_VALIDATION = 1000
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

def run_model(train_data, train_labels, test_data, test_labels, model):
	model.fit(train_data, train_labels, epochs=9, batch_size=512)

	print("results: ", model.evaluate(test_data, test_labels))
	print("prediction for each test sample: ",model.predict(test_data)[0])

def reverse_word_index(word_index):
	return dict([(value, key) for (key, value) in word_index.items()])

# using larger dimensions needs more RAM, or MemoryError is thrown
# creates a matrix with shape(len(sequences),  dimension).
# One-Hot encoding
def vectorize_sequences(sequences, dimension):
	results = np.zeros(shape=(len(sequences), dimension), dtype=float)
	for i, sequence in enumerate(sequences):
		results[i, sequence] = 1.
	return results

def main():
	# to fix: "ValueError: Object arrays cannot be loaded when allow_pickle=False"
	# see: https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa

	# save np.load
	np_load_old = np.load

	# modify the default parameters of np.load
	np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

	(train_data, train_labels), (test_data, test_labels) =  reuters.load_data(num_words=_NUM_WORDS)

	# restore np.load for future normal usage
	np.load = np_load_old

	# convert word index to words
	_reverse_word_index = reverse_word_index(reuters.get_word_index())
	decoded_review = ' '.join([_reverse_word_index.get(i - 3, '?') for i in train_data[0]])
	print(decoded_review)

	binary_train_data = vectorize_sequences(train_data, _NUM_WORDS) # TODO could we reuse to_categorical() ?
	binary_test_data = vectorize_sequences(test_data, _NUM_WORDS)
	# binary data contains a list of 0 and 1 only
	# print("binary_train_data[0]: ", binary_train_data[0])

	one_hot_train_labels = to_categorical(train_labels)
	one_hot_test_labels = to_categorical(test_labels)

	# Create Model
	model =  models.Sequential()
	model.add(layers.Dense(64, activation='relu',  input_shape=(_NUM_WORDS,)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(46, activation='softmax'))

	model.compile(optimizer='rmsprop',
			loss='categorical_crossentropy',
			metrics=['accuracy'])

	run_validation_model(binary_train_data, one_hot_train_labels, binary_test_data, one_hot_test_labels, model)
	# run_model(binary_train_data, one_hot_train_labels, binary_test_data, one_hot_test_labels, model)
    
if __name__ == '__main__':
	main()

