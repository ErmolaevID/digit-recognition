import numpy
from tensorflow import keras

from neural_network_settings import image_size


def test_neural_network(neural_network, image_file):
    image = keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size), color_mode='grayscale')
    image_data = numpy.expand_dims(image, axis=0)
    image_data = 1 - image_data / 255.0
    image_data = image_data.reshape((1, 28, 28, 1))

    return neural_network.predict_classes([image_data])[0]
