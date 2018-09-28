import tensorflow.keras.backend as K

def get_weighted_binary_crossentropy(pos_weight=1, neg_weight=1):

    def weighted_binary_crossentropy(y_true, y_pred):
        binary_crossentropy = K.binary_crossentropy(y_true, y_pred)

        weights = y_true * pos_weight + (1. - y_true) * neg_weight

        weighted_binary_crossentropy_vector = weights * binary_crossentropy

        return K.mean(weighted_binary_crossentropy_vector)

    return weighted_binary_crossentropy