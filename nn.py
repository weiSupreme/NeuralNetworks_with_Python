import numpy as np
import random
import time

class NeuralNetwork:
    def __init__(self, parameters):
        self.W = []
        self.global_step = 0
        self.max_step = 0

        self.layers = parameters['layers']
        self.batch = parameters['batch']
        self.activation_fn = parameters['activation_fn']

        # learn_rate
        lr = parameters['learning_rate']
        self.learning_rate = lr['initial_rate']
        self.initial_learning_rate = lr['initial_rate']
        self.end_learning_rate = lr['end_rate']
        self.learning_rate_power = lr['power']


        for i in range(0, len(self.layers) - 2):
            w = np.random.randn(self.layers[i] + 1, self.layers[i + 1] + 1)
            self.W.append(w / np.sqrt(self.layers[i]))

        w = np.random.randn(self.layers[-2] + 1, self.layers[-1])
        self.W.append(w / np.sqrt(self.layers[-2]))

    def __repr__(self):
        return "NeuralNetwork:{}".format('-'.join(str(l) for l in self.layers))

    def activation(self, x, name=None):
        if name == None:
            name = self.activation_fn
        if name == 'sigmoid':
            return 1.0 / (1 + np.exp(-x))
        elif name == 'relu':
            return x * (x>0)
        else:
            raise Exception('There is no {} activation function'.format(name))

    def activation_deriv(self, x, name=None):
        if name == None:
            name = self.activation_fn
        if name == 'sigmoid':
            return x * (1 - x)
        elif name == 'relu':
            return 1. * (x>0)
        else:
            raise Exception('There is no {} activation function'.format(name))

    def softmax(self, x):
        x_exp = np.exp(x - np.max(x))
        sum=np.sum(x_exp, axis=1)
        for i in range(x_exp.shape[0]):
            x_exp[i]=x_exp[i]/sum[i]
        return x_exp

    def data_gen(self, X, Y):
        num = len(X)
        num_infact = 6000
        shuffle = random.sample(range(num), num_infact)
        count = 0
        while count<num_infact:
            yield X[shuffle[count:count+self.batch]], Y[shuffle[count:count+self.batch]], True
            count += self.batch
        yield X[shuffle[count-self.batch:num_infact]], Y[shuffle[count-self.batch:num_infact]], False

    def train(self, X, Y, epochs=100, displayUpdate=10):
        self.max_step = epochs * len(X) // self.batch
        X = np.append(X,np.ones((X.shape[0],1)),axis=1)

        for epoch in range(0, epochs):
            dg = self.data_gen(X, Y)
            flag = True
            start = time.clock()
            while flag:
                x, y, flag = next(dg)
                A = self.forward(x)
                self.backward(A, np.array(y))
                self.global_step += 1
            end = time.clock()
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, Y)
                print('[INFO] epoch={}, loss={:.7f}'.format(epoch + 1, loss))
                #print('[INFO] epoch={}, loss={:.7f}, time={:.3f}'.format(epoch + 1, loss, end-start))

    def forward(self, x):
        A = [np.atleast_2d(x)]
        for layer in range(0, len(self.W) - 1):
            net = A[layer].dot(self.W[layer])
            out = self.activation(net)
            A.append(out)
        layer = len(self.W) - 1
        net = A[layer].dot(self.W[layer])
        out = self.softmax(net)
        A.append(out)
        return A

    def backward(self, A, y):
        D = A[-1]
        D[np.arange(y.shape[0]), y.astype('int')] -= 1
        D = [D]
        for layer in range(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.activation_deriv(A[layer])
            D.append(delta)

        D = D[::-1]

        # update lr nad W
        self.learning_rate = self.end_learning_rate + (self.initial_learning_rate - self.end_learning_rate) * (1 - float(self.global_step/self.max_step)) ** self.learning_rate_power
        for layer in range(len(self.W)):
            self.W[layer] += -self.learning_rate * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)
        if addBias:
            p = np.append(p, np.ones((p.shape[0],1)),axis=1)

        for layer in range(len(self.W) - 1):
            p = self.activation(np.dot(p, self.W[layer]))
        p = self.softmax(np.dot(p, self.W[len(self.W) - 1]))
        return p

    def calculate_loss(self, X, target):
        target = np.atleast_2d(target)
        predictions = self.predict(X, addBias=False)
        loss = -np.log(predictions[np.arange(predictions.shape[0]),target.astype('int')])
        return np.sum(loss)/loss.shape[1]