import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report
from nn import NeuralNetwork
import random


# load mnist and normalize
mnist =fetch_mldata("MNIST original", data_home='./mnist')
x, y = mnist['data'], mnist['target']
x=x.astype('float')
x=(x-x.min())/(x.max()-x.min())

# split dataset
trainX, testX, trainY, testY = x[:60000], x[60000:], y[:60000], y[60000:]

# parameters
lr = {'initial_rate': 0.01, \
          'end_rate': 0.0001, \
            'momentum': 0.9}
params ={'layers': [trainX.shape[1], 100, 10], \
             'learning_rate': lr, \
               'batch': 32, \
                 'activation_fn': 'sigmoid'}

net = NeuralNetwork(params)
print("[INFO] {}".format(net))

net.train(trainX, trainY, epochs=20, displayUpdate=1)

predictions = net.predict(testX)
print(classification_report(testY, predictions.argmax(axis=1)))
