# ===========> A module to define a fully connected deep learning model <===========

# The model is defined with numpy only
# No prebuilt ML APIs, just pure mathematics


import numpy as np
from time import sleep

sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_der = lambda x: sigmoid(x) * (1 - sigmoid(x))
ReLU = lambda x: np.maximum(0, x)
ReLU_der = lambda x: x > 0


class NeuralNetwork:
    def __init__(self, shape: tuple):
        self.layers = len(shape)
        assert self.layers > 1
        self.shape = shape
        for l in shape: assert l > 0
        self.weights = [
            np.array([2 * np.random.random(shape[l - 1]) - 1 for _ in range(shape[l])])
            for l in range(1, self.layers)
        ]
        self.biases = [np.zeros(l) for l in shape[1:]]
        self.weighted_sums = [np.zeros(l) for l in shape[1:]]
        self.activation_functions = [sigmoid for _ in range(self.layers - 2)] + [sigmoid]
        self.af_ders = [sigmoid_der for _ in range(self.layers - 2)] + [sigmoid_der]
        self.activations = [np.zeros(l) for l in shape]
        self.deltas = [None for l in shape[1:]]
        self.visualised = False

    def _inspect(self):
        print("=============NeuralNetwork===============")
        print(f"Shape: {self.shape}")
        print(f"Weights: {self.weights}")
        print(f"Biases: {self.biases}")
        print(f"Activations: {self.activations}")

    def forward_prop(self, X):
        self.activations[0][:len(X)] = X[:self.shape[0]]
        for l in range(self.layers - 1):
            self.weighted_sums[l] = self.weights[l] @ self.activations[l] + self.biases[l]
            self.activations[l + 1] = self.activation_functions[l](self.weighted_sums[l])

    def backprop(self, X, Y, lr):
        self.deltas[-1] = (Y - self.activations[-1]) * self.af_ders[-1](self.weighted_sums[-1])
        for l in range(self.layers - 2, 0, -1):
            self.deltas[l - 1] = self.weights[l].T @ self.deltas[l] * self.af_ders[l - 1](self.weighted_sums[l - 1])
        for l in range(self.layers - 1):
            for j in range(self.shape[l + 1]):
                self.weights[l][j] += lr * self.activations[l] * self.deltas[l][j]
            self.biases[l] += self.deltas[l]

    def train(self, X, Y, lr, epochs, classified = False):
        for e in range(epochs):
            if not e % 1000 and not self.visualised: self.test(X, Y, classified)
            i = np.random.randint(len(X))
            self.forward_prop(X[i])
            if self.visualised: self.update_screen()
            self.backprop(X[i], Y[i], lr)

    def test(self, X, Y, classified = False):
        while self.visualised:
            i = np.random.randint(len(X))
            self.forward_prop(X[i])
            self.update_screen()
            if not classified: print(self.activations[-1], Y[i])
            else: print(np.argmax(self.activations[-1]), np.argmax(Y[i]))
            sleep(0.2)
        print()
        if classified: correct_predictions = predictions = 0
        for i in range(len(X)):
            self.forward_prop(X[i])
            if not classified: print(self.activations[-1], Y[i])
            else:
                if np.argmax(self.activations[-1]) == np.argmax(Y[i]): correct_predictions += 1
                predictions += 1
        if classified: print(f"Accuracy: {round(correct_predictions / predictions * 100, 1)}%")

    from Visualisation import init_visualisation, update_screen


if __name__ == "__main__":
    nn = NeuralNetwork((2, 4, 1))
    
    X = np.array([
        [1, 0],
        [0, 1],
        [1, 1],
        [0, 0]
    ])

    Y = np.array([
        [1],
        [1],
        [0],
        [0]
    ])

    nn.init_visualisation(10)
    nn.train(X, Y, 0.4, 10000)
    nn._inspect()
    nn.test(X, Y)