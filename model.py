from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import Input, Dropout, Dense
from tensorflow.python.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import keras_metrics
import my_losses






def get_model(alpha=1, depth_multiplier=1, pooling='avg', reg=0, lr = 0.00001):

    img_input = Input(shape=(224, 224, 1))

    if pooling == 'avg':
        x = GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = GlobalMaxPooling2D()(x)

    x = Dropout(0.5)(x)
    x = Dense(512)(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)


    # Create model.
    model = Model(img_input, x, name='mobilenet_%0.2f_%s' % (alpha, 224))
    optimizer = Adam(lr)
    loss = my_losses.get_weighted_binary_crossentropy(0.55, 0.45)
    model.compile(optimizer=optimizer, loss=loss,
                  metrics=[keras_metrics.precision(), keras_metrics.recall(), keras_metrics.f1_score()])

    return model