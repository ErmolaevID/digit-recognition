import numpy
from tensorflow.python.keras.datasets.mnist import load_data
from tensorflow.python.keras.utils.np_utils import to_categorical

from neural_network_settings import classes, image_size, channels


def train_neural_network(neural_network):
    (train_digits, train_labels), (test_digits, test_labels) = load_data()

    train_data = numpy.reshape(train_digits, (train_digits.shape[0], image_size, image_size, channels))
    train_data = train_data.astype('float32') / 255.0

    train_cat = to_categorical(train_labels, classes)

    value = numpy.reshape(test_digits, (test_digits.shape[0], image_size, image_size, channels))
    value = value.astype('float32') / 255.0
    value_cat = to_categorical(test_labels, classes)

    neural_network.fit(train_data, train_cat, epochs=8, batch_size=64, validation_data=(value, value_cat))

    return neural_network
