import numpy as np

class Perceptron:
    def __init__(self, num_features):
        self.weights = np.array([0, 0, 0])
        self.bias = 1

    def predict(self, X):
        activation = np.dot(X, self.weights) + self.bias
        return 1 if activation >= 0 else 0

    def train(self, X_train, y_train, learning_rate=1, epochs=10):
        for _ in range(epochs):
            for X, y in zip(X_train, y_train):
                prediction = self.predict(X)
                error = y - prediction
                self.weights += learning_rate * error * X
                self.bias += learning_rate * error

X_train = np.array([[4, 3, 3], [2, -2, 3], [1, 0, -3], [4, 2, 2]])
y_train = np.array([0, 1, 1, 0])

perceptron = Perceptron(num_features=X_train.shape[1])
perceptron.train(X_train, y_train)
print('weights:', perceptron.weights)
print('bias:', perceptron.bias)

X_test = np.array([[2, 4, 5]])
for X in X_test:
    print(f"Prediction for {X}: Class {perceptron.predict(X)}")