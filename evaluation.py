# trains a separate model with all train data/labels and evaluates the quality with the test data/labels
def evaluate(model, train_data, train_labels, test_data, test_labels, epochs, batch_size, verbose):
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              verbose=verbose)

    result = model.evaluate(test_data, test_labels)
    print("evaluation of the trained model: " + str(result[1]))

    # show prediction samples
    # print("prediction for each test sample: ",model.predict(test_data))