import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.metrics import classification_report
from nn import NeuralNetwork


# load mnist and normalize
mnist =fetch_mldata("MNIST original", data_home='./mnist')
x, y = mnist['data'], mnist['target']
x=x.astype('float')
x=(x-x.min())/(x.max()-x.min())

trainX, testX, trainY_, testY_ = x[:60000], x[60000:], y[:60000], y[60000:]
shuffle_idx = np.random.permutation(60000)
trainX, trainY_ = trainX[shuffle_idx], trainY_[shuffle_idx]

# deal with y
trainY=np.zeros((trainX.shape[0], 10))
trainY[np.arange(trainX.shape[0]),trainY_.astype('int')]=1.0

testY=np.zeros((testX.shape[0], 10))
testY[np.arange(testX.shape[0]),testY_.astype('int')]=1.0

# train
net = NeuralNetwork([trainX.shape[1], 32, 16, 10], alpha=0.01, activation_fn='relu')
print("[INFO] {}".format(net))

net.train(trainX, trainY, epochs=20, displayUpdate=1)

predictions = net.predict(testX)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1)))
