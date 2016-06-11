
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np
import pylab as pl
import net, util

if __name__ == '__main__':
  np.random.seed(1)
  X = np.random.rand(20, 5)
  y = (X.sum(axis=1)**3 + 1)[:,np.newaxis]

  model = net.Net([net.Dense(5,1), net.Relu()])

  W = model.layers[0].W.copy()
  b = model.layers[0].b.copy()

  model.fit(X, y, loss=util.MSE(), optimizer=util.SGD(0.1, 0), n_epoch=5, batch_size=10)
  y_hat = model.predict(X)

  model1 = Sequential([
      Dense(1, input_dim=5, weights=[W.T, b]),
      Activation('relu')
    ])
  sgd = SGD(lr=0.1, decay=0, momentum=0, nesterov=False)
  model1.compile(loss='mse', optimizer=sgd)

  model1.fit(X, y, nb_epoch=5, batch_size=10, shuffle=False)
  y_hat1 = model1.predict(X)

  print(np.allclose(y_hat, y_hat1))
  print(model.layers[0].W, model.layers[0].b)
  print(model1.get_weights())
  pl.plot(y_hat, '--o')
  pl.plot(y_hat1, '--*')
  pl.show()
