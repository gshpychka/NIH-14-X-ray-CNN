from keras.applications.mobilenetv2 import MobileNetV2
from keras.models import Sequential
from keras.layers import Dropout, Dense
from keras.models import Model
from keras.optimizers import Adam
import keras
import keras_metrics


def get_model(alpha=1, depth_multiplier=1, pooling='avg', lr = 0.00001):

    base_mobilenetv2_model = MobileNetV2(alpha=alpha, depth_multiplier=depth_multiplier, input_shape=(224, 224, 1), include_top=False, weights=None,
                                                    classes=1, pooling=pooling)

    top_model = Sequential()
    top_model.add(Dense(512))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))
    # Create model.
    model = Model(inputs=base_mobilenetv2_model.input, outputs=top_model(base_mobilenetv2_model.output))
    optimizer = Adam(lr=lr)
    model.compile(optimizer=optimizer, loss='binary_crossentropy',
                  metrics=[keras.metrics.binary_accuracy, keras_metrics.precision()])

    return model