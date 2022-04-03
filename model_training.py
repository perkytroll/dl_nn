import os
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from tensorflow.python.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as mpl
from tensorflow.python.keras.optimizer_v2.adamax import Adamax


class ModelTraining:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test

    def build_architecture(self):
        model = keras.Sequential()
        model.add(layers.LSTM(512, input_shape=(self.x_train.shape[1], self.x_train.shape[2]),
                              return_sequences=True, recurrent_dropout=0.3))
        model.add(layers.Dropout(0.1))
        model.add(layers.LSTM(256))
        model.add(layers.Dense(256))
        model.add(layers.Dropout(0.1))
        # model.add(keras.layers.Flatten())
        print(self.y_train.shape[1])
        model.add(layers.Dense(units=self.y_train.shape[1]))
        model.add(layers.Activation('softmax'))
        return model

    @staticmethod
    def compile_model(model_architecture):
        optimizer = Adamax(learning_rate=0.01)
        model_architecture.compile(optimizer=optimizer,
                                   loss='categorical_crossentropy')
        return model_architecture

    def model_training(self, compiled_model):
        filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
        checkpoint = ModelCheckpoint(
            filepath, monitor='loss',
            verbose=0,
            save_best_only=True,
            mode='min'
        )
        history = compiled_model.fit(self.x_train, self.y_train, epochs=1, verbose=1, batch_size=128,
                                     callbacks=[checkpoint])
        return history, compiled_model

    @staticmethod
    def best_weight_file():
        epochs = []
        loss = []
        for file_itr in os.listdir('./'):
            file_name, file_extension = os.path.splitext(file_itr)
            if file_extension == '.hdf5':
                splits = file_name.split('-')
                bigger_index = splits.index('bigger')
                epochs.append(splits[bigger_index - 2])
                loss.append(float(splits[bigger_index - 1]))
        min_loss = min(loss)
        min_loss_epoch = epochs[loss.index(min_loss)]
        return 'weights-improvement-' + str(min_loss_epoch) + '-' + str(min_loss) + '-bigger.hdf5'

    def weight_updated_model(self, model_architecture):
        model_architecture.load_weights(self.best_weight_file())
        return model_architecture

    @staticmethod
    def plot_training_results(history):
        mpl.plot(history.history['loss'], color='green', linewidth=3)
        mpl.xlabel('Epochs')
        mpl.ylabel('Accuracy')
        mpl.title('Training Loss trend')
        mpl.legend(['Training'], loc='upper right')
        mpl.show()
