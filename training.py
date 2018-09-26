import tensorflow.keras as keras
from tensorflow.keras.utils import HDF5Matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np

datapath = 'data/data_new.h5'


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

def load_data(train_start, n_train, test_start, n_test):
    X_train = HDF5Matrix(datapath, 'X', train_start, train_start+n_train)
    y_train = HDF5Matrix(datapath, 'Any Finding', train_start, train_start+n_train)
    X_test = HDF5Matrix(datapath, 'X', test_start, test_start+n_test)
    y_test = HDF5Matrix(datapath, 'Any Finding', test_start, test_start+n_test)
    return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, datagen, epochs=1, batch_size=32):
    for e in range(epochs):
        print('Currently in epoch %d out of %d' % (e+1, epochs))
        for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=batch_size):
            loss = model.train(X_batch, y_batch)
            print("Current loss: %d" % loss)