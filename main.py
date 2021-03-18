import mnist_loader
from net import Network

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10])
    net.stochastic_gradient_descent(training_data, 10, 10, 3.0, test_data=test_data)