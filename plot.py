import matplotlib
import os

matplotlib.use('Agg')

import matplotlib.pyplot as plt


class Plot:
    def __init__(self, history, title, project_directory):
        self.history_dict = history.history
        self.title = title

        # create output directory
        self.directory = 'outputs/' + project_directory + '/'
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        print(history.history.keys())

    def plot(self):
        self.plot_loss_result()
        self.plot_accuracy_result()

    def plot_loss_result(self):
        loss_values = self.history_dict['loss']
        validation_loss_values = self.history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)

        plt.clf()
        plt.plot(epochs, loss_values, 'bo', label="Loss Training")
        plt.plot(epochs, validation_loss_values, 'b', label='Loss Validation')
        plt.title('Loss function: ' + self.title)
        plt.xlabel('Epochs')
        plt.ylabel('Value loss function')
        plt.legend()
        plt.savefig(self.directory + 'loss.png')

    def plot_accuracy_result(self):
        accuracy_values = self.history_dict['acc']
        validation_accuracy_values = self.history_dict['val_acc']
        epochs = range(1, len(accuracy_values) + 1)

        plt.clf()
        plt.plot(epochs, accuracy_values, 'bo', label='Accuracy Traning')
        plt.plot(epochs, validation_accuracy_values, 'b', label='Accuracy Validation')
        plt.title('Accuracy Results: ' + self.title)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy value')
        plt.legend()
        plt.savefig(self.directory + 'accuracy.png')
