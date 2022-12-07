from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import MaxPooling2D, Conv2D, Flatten, Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam

from neural_network_settings import image_size, channels, classes


def create_neural_network():
    neural_network = Sequential()
    neural_network.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, channels)))
    neural_network.add(MaxPooling2D(pool_size=(2, 2)))
    neural_network.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    neural_network.add(MaxPooling2D(pool_size=(2, 2)))
    neural_network.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    neural_network.add(MaxPooling2D(pool_size=(2, 2)))
    neural_network.add(Flatten())
    neural_network.add(Dense(128, activation='relu'))
    neural_network.add(Dense(classes, activation='softmax'))
    neural_network.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return neural_network
