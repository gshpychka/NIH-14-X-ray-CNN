import tensorflow.keras as keras
from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
from model import get_model

datapath = 'data/data_new.h5'

def load_data(train_start=0, n_train=50000, test_start=80000, n_test=1000):
    X_train = HDF5Matrix(datapath, 'X', train_start, train_start+n_train)
    y_train = HDF5Matrix(datapath, 'labels/Any Finding', train_start, train_start+n_train)
    X_test = HDF5Matrix(datapath, 'X', test_start, test_start+n_test)
    y_test = HDF5Matrix(datapath, 'labels/Any Finding', test_start, test_start+n_test)
    return X_train, y_train, X_test, y_test


def train_model(model, X_train, y_train, X_test, y_test, datagen, epochs=1, batch_size=32):
    model.fit_generator(datagen.flow(X_train, y_train, shuffle=False, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / batch_size,
                        epochs=epochs,
                        validation_data=(X_test,y_test))

if __name__ == '__main__':
    model = get_model()
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rotation_range=3
    )
    X_representative_set = HDF5Matrix(datapath, 'X', end=5000)
    print("Fitting mean and std on a representative set...")
    datagen.fit(X_representative_set)

    X_train, y_train, X_test, y_test = load_data()
    train_model(model, X_train, y_train, X_test, y_test, datagen, batch_size=5)