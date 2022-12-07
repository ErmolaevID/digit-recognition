from tensorflow.python.keras.models import load_model

from create_neural_network import create_neural_network
from test_neural_network import test_neural_network
from train_neural_network import train_neural_network

model_for_train = create_neural_network()

train_neural_network(model_for_train)

model_for_train.save('trained_neural_network.h5')

neural_network_for_test = load_model('trained_neural_network.h5')

print(test_neural_network(neural_network_for_test, 'images_for_tests/0.png'))
print(test_neural_network(neural_network_for_test, 'images_for_tests/1.png'))
print(test_neural_network(neural_network_for_test, 'images_for_tests/2.png'))
print(test_neural_network(neural_network_for_test, 'images_for_tests/3.png'))
print(test_neural_network(neural_network_for_test, 'images_for_tests/4.png'))
print(test_neural_network(neural_network_for_test, 'images_for_tests/5.png'))
print(test_neural_network(neural_network_for_test, 'images_for_tests/6.png'))
print(test_neural_network(neural_network_for_test, 'images_for_tests/7.png'))
print(test_neural_network(neural_network_for_test, 'images_for_tests/8.png'))
print(test_neural_network(neural_network_for_test, 'images_for_tests/9.png'))
