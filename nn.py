import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha=0.1, activation_fn='sigmoid'):
        self.W = []
        self.layers = layers
        self.alpha = alpha
        self.activation_fn = activation_fn

        for i in range(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        return "NeuralNetwork:{}".format('-'.join(str(l) for l in self.layers))

    def activation(self, x, name = None):
        if name == None:
            name = self.activation_fn
        if name == 'sigmoid':
            return 1.0 / (1 + np.exp(-x))
        elif name == 'relu':
            return x * (x>0)
        else:
            raise Exception('There is no {} activation function'.format(name))

    def activation_deriv(self, x, name = None):
        if name == None:
            name = self.activation_fn
        if name == 'sigmoid':
            return x * (1 - x)
        elif name == 'relu':
            return 1. * (x>0)
        else:
            raise Exception('There is no {} activation function'.format(name))


    def train(self, X, y, epochs=1000, displayUpdate=100):
        X = np.append(X,np.ones((X.shape[0],1)),axis=1)

        for epoch in range(0, epochs):
            for (x, target) in zip(X, y):
                A = self.forward(x)
                self.backward(A, target)
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print('[INFO] epoch={}, loss={:.7f}'.format(epoch + 1, loss))

    def forward(self, x):
        A = [np.atleast_2d(x)]
        for layer in range(0, len(self.W) - 1):
            net = A[layer].dot(self.W[layer])
            out = self.activation(net)
            A.append(out)
        layer = len(self.W) - 1
        net = A[layer].dot(self.W[layer])
        out = self.activation(net, name='sigmoid')
        A.append(out)
        return A

    def backward(self,A,y):
        error = A[-1] - y
        D = [error * self.activation_deriv(A[-1], name='sigmoid')]
        for layer in range(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.activation_deriv(A[layer])
            D.append(delta)

        D = D[::-1]

        # update W
        for layer in range(len(self.W)):
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        if addBias:
            p = np.append(p, np.ones((p.shape[0],1)),axis=1)

        for layer in range(len(self.W) - 1):
            p = self.activation(np.dot(p, self.W[layer]))
        layer = len(self.W) - 1
        p = self.activation(np.dot(p, self.W[layer]), name='sigmoid')
        return p

    def calculate_loss(self, X, target):
        target = np.atleast_2d(target)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - target)**2)
        return loss / predictions.shape[0]