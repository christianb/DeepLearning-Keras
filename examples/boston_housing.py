# Boston Housing 
# 404 trainings samples
# 102 test samples

from keras.datasets import boston_housing
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def build_model(train_data):
	model =  models.Sequential()
	model.add(layers.Dense(64, activation='relu',  input_shape=(train_data.shape[1],)))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(1))

	model.compile(optimizer='rmsprop',
			loss='mse',
			metrics=['mae'])
	return model

def main():
	# to fix: "ValueError: Object arrays cannot be loaded when allow_pickle=False"
	# see: https://stackoverflow.com/questions/55890813/how-to-fix-object-arrays-cannot-be-loaded-when-allow-pickle-false-for-imdb-loa

	# save np.load
	np_load_old = np.load

	# modify the default parameters of np.load
	np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

	(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

	# restore np.load for future normal usage
	np.load = np_load_old

	# normalize data
	mean = train_data.mean(axis=0)
	train_data -= mean
	std = train_data.std(axis=0)
	train_data /= std

	test_data -= mean # TODO could we calculate the mean and std from test_data itself?
	test_data /= std

	# k-cross validation
	k=4
	num_val_samples = len(train_data) // k 
	print('num_val_samples: ', num_val_samples)
	num_epochs = 50
	all_scores = []
	all_mae_histories = []

	for i in range(k):
		print('Run #', i)
		val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
		val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

		partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0)
		partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

		model = build_model(train_data)
		
		history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets), epochs=num_epochs, batch_size=1, verbose=0)
		
		val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
		
		mae_history = history.history['val_mean_absolute_error']
		all_scores.append(val_mae)
		all_mae_histories.append(mae_history)

	print('all_scores: ', all_scores)
	print('mean all_scores: ', np.mean(all_scores))

	average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

	# plot
	plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
	plt.xlabel('Epochs')
	plt.ylabel('Mean absolut error - Validation')
	plt.show()
if __name__ == '__main__':
	main()

