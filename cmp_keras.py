
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

  model = net.Net([net.Dense(5,10), 
                   net.Relu(),
                   net.Dense(10,100),
                   net.Sigmoid(),
                   net.Dense(100,1),
                   net.Linear()
                 ])

  W = model.layers[0].W.copy()
  b = model.layers[0].b.copy()
  W1 = model.layers[2].W.copy()
  b1 = model.layers[2].b.copy()
  W2 = model.layers[4].W.copy()
  b2 = model.layers[4].b.copy()

  model.fit(X, y, loss=util.MSE(), optimizer=util.SGD(0.01, 0), n_epoch=5, batch_size=10)
  y_hat = model.predict(X)

  model1 = Sequential([
      Dense(10, input_dim=5, weights=[W.T, b]),
      Activation('relu'),
      Dense(100, weights=[W1.T, b1]),
      Activation('sigmoid'),
      Dense(1, weights=[W2.T, b2]),
      Activation('linear')
    ])
  sgd = SGD(lr=0.01, decay=0, momentum=0, nesterov=False)
  model1.compile(loss='mse', optimizer=sgd)

  model1.fit(X, y, nb_epoch=5, batch_size=10, shuffle=False)
  y_hat1 = model1.predict(X)

  print(np.allclose(y_hat, y_hat1))
  pl.plot(y_hat, '--o')
  pl.plot(y_hat1, '--*')
  pl.show()
