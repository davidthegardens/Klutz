from keras.callbacks import Callback
from keras.models import save_model

class CustomSaver(Callback):
    def __init__(self, threshold=None):
        super(CustomSaver, self).__init__()
        self.threshold = threshold
        self.best_accuracy = 0

    def on_train_batch_end(self, batch, logs=None):
        accuracy = logs.get('accuracy')
        if self.threshold is not None:
            if self.threshold <= accuracy and accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                save_model(self.model, 'best_model_unstratified.h5')
                print(f"\nModel saved with accuracy: {accuracy}")