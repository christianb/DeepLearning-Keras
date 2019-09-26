import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt


class Plot:
    def __init__(self, history):
        self.history = history

    def plot(self):
        self.plot_loss_result()
        # self.plot_accuracy_result()

    def plot_loss_result(self):
        history_dict = self.history.history
        loss_values = history_dict['loss']
        validation_loss_values = history_dict['val_loss']
        epochs = range(1, len(loss_values) + 1)

        plt.plot(epochs, loss_values, 'bo', label="Loss Training")
        plt.plot(epochs, validation_loss_values, 'b', label='Loss Validation')
        plt.title('Loss function results')
        plt.xlabel('Epochs')
        plt.ylabel('Value loss function')
        plt.legend()
        plt.savefig('loss.png')

    def plot_accuracy_result(self):
        history_dict = self.history.history

        acc_values = history_dict['acc']
        validation_acc_values = history_dict['val_acc']
        epochs = range(1, len(acc_values) + 1)

        plt.plot(epochs, acc_values, 'bo', label='Accuracy Traning')
        plt.plot(epochs, validation_acc_values, 'b', label='Accuracy Validation')
        plt.title('Accuracy Results')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy value')
        plt.legend()
        plt.savefig('accuracy.png')
