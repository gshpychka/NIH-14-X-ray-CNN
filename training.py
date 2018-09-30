import tensorflow.keras as keras
from keras.utils import HDF5Matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from model import get_model
from keras.applications.mobilenetv2 import preprocess_input

datapath = 'data/data_new.h5'


def get_batches(X, y, batch_size):
    while True:
        for i in range(int(len(X) / batch_size)):
            start = i*batch_size
            end = (i+1)*batch_size
            if end > len(X):
                end = len(X)
            yield X[start:end], y[start:end]


def my_datagen(X, y, batch_size, keras_datagen=None):
    if keras_datagen is None:
        use_keras_datagen = False
    else:
        use_keras_datagen = True
    while True:
        for X_batch, y_batch in get_batches(X, y, batch_size):
            if use_keras_datagen:
                yield next(keras_datagen.flow(
                        X_batch,
                        y_batch,
                        shuffle=False,
                        batch_size=batch_size))
            else:
                yield preprocess_input(X_batch), y_batch


def load_data(train_start=0, n_train=100000, test_start=100000, n_test=5000, label='Infiltration'):

    X_train = HDF5Matrix(datapath, 'X', train_start, train_start+n_train)
    y_train = HDF5Matrix(datapath, 'labels/' + label, train_start, train_start+n_train)
    X_test = HDF5Matrix(datapath, 'X', test_start, test_start+n_test)
    y_test = HDF5Matrix(datapath, 'labels/' + label, test_start, test_start+n_test)
    return X_train, y_train, X_test, y_test


def train_model(model,
                label,
                train_start=0,
                n_train=100000,
                test_start=100000,
                n_test=1000,
                epochs=1,
                steps_per_epoch=None,
                batch_size=32):
    print('Fetching data')
    X_train, y_train, X_test, y_test = load_data(train_start, n_train, test_start, n_test, label)
    print("Starting training.")

    if steps_per_epoch is None:
        steps_per_epoch = len(X_train) / batch_size
    return model.fit_generator(my_datagen(X_train, y_train, batch_size),
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=(X_test,y_test),
                        use_multiprocessing=False,
                        shuffle=False,
                        class_weight={0: 22, 1: 78})

if __name__ == '__main__':
    #model = get_model(reg=0, lr=0.001)
    X_representative_set = HDF5Matrix(datapath, 'X', end=1000)
    X_train, y_train, X_test, y_test = load_data()
    #history = train_model(model, X_train, y_train, X_test, y_test,
    #                      batch_size=10,
    #                      epochs=5
    #                      )