import tensorflow.keras as keras
from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from model import get_model

datapath = 'data/data_new.h5'


def get_batches(X, y, batch_size):
    while True:
        for i in range(int(len(X) / batch_size)):
            start = i*batch_size
            end = (i+1)*batch_size
            if end > len(X):
                end = len(X)
            yield X[start:end], y[start:end]


def my_datagen(X_train, y_train, batch_size, keras_datagen):
    while True:
        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            for X_mini, y_mini in keras_datagen.flow(X_batch,
                                               y_batch,
                                               shuffle=False,
                                               batch_size=batch_size):
                yield X_mini, y_mini


def load_data(train_start=0, n_train=100000, test_start=100000, n_test=1000):
    X_train = HDF5Matrix(datapath, 'X', train_start, train_start+n_train)
    y_train = HDF5Matrix(datapath, 'labels/Pneumonia', train_start, train_start+n_train)
    X_test = HDF5Matrix(datapath, 'X', test_start, test_start+n_test)
    y_test = HDF5Matrix(datapath, 'labels/Pneumonia', test_start, test_start+n_test)
    return X_train, y_train, X_test, y_test


def train_model(model, X_train, y_train, X_test, y_test, keras_datagen, epochs=1, batch_size=32):
    print("Starting training.")
    return model.fit_generator(my_datagen(X_train, y_train, batch_size, keras_datagen),
                        steps_per_epoch=200,
                        epochs=epochs,
                        validation_data=(X_test,y_test),
                        use_multiprocessing=False,
                        shuffle=False)

if __name__ == '__main__':
    model = get_model()
    keras_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=3
    )
    X_representative_set = HDF5Matrix(datapath, 'X', end=5000)
    print("Fitting mean and std on a representative set...")
    keras_datagen.fit(X_representative_set)
    print("Done.")

    X_train, y_train, X_test, y_test = load_data()
    history = train_model(model, X_train, y_train, X_test, y_test, keras_datagen, batch_size=10, epochs=5)