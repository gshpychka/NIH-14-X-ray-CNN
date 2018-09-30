
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import Input, Dropout, Dense
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import keras_metrics
import my_losses






def get_model(alpha=1, depth_multiplier=1, pooling='avg', lr = 0.00001):

    base_mobilenet_model = MobileNet(input_shape=(224, 224, 1),
                                     include_top=False, weights=None)
    model = Sequential()
    model.add(base_mobilenet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # Create model.

    optimizer = Adam(lr)
    loss = my_losses.get_weighted_binary_crossentropy(0.55, 0.45)
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=[keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])

    return model