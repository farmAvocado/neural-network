
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD
import numpy as np
import pylab as pl
import net

if __name__ == '__main__':
  np.random.seed(1)
  X = np.random.rand(80, 5)
  y = (X.sum(axis=1)**3 + 1)[:,np.newaxis]

  model2 = net.Net([net.Dense(5,1), net.Linear()])

  W = model2.layers[0].W.copy()
  b = model2.layers[0].b.copy()

  model2.train(X, y, n_epochs=50, batch_size=20, lr=0.1)
  y_hat2 = model2.forward(X)

  model = Sequential([
      Dense(1, input_dim=5),  #, weights=[W.T, b]),
      Activation('linear')
    ])
  sgd = SGD(lr=0.1, decay=0, momentum=0, nesterov=False)
  model.compile(loss='mse', optimizer=sgd)

  model.fit(X, y, nb_epoch=50, batch_size=20)
  y_hat = model.predict(X)

  pl.plot(y, '-^')
  pl.plot(y_hat, '--o')
  pl.plot(y_hat2, '--*')
  pl.show()
