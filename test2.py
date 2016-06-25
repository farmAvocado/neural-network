
import numpy as np
import pylab as pl
import net2

pl.style.use('ggplot')

def test_regression():
  X = np.random.rand(50, 2)
  y = X.sum(axis=1, keepdims=True)**2
  X = net2.standardize(X)
  model = net2.Net([net2.Dense(X.shape[1],10), 
                   net2.Relu(),
                   net2.Dense(10,1),
                   net2.Relu(),
                 ])
  model.fit(X, y, loss=net2.MSE(), optimizer=net2.SGD(0.01, 1.0), n_epoch=100, batch_size=1)
  y_hat = model.predict(X)

  pl.plot(y, 'b->')
  pl.plot(y_hat, 'm--o')
  pl.show()

def test_classification():
  X = np.random.rand(50, 2)
  X[:20,:] += 1
  y = np.zeros((50,2), dtype='int')
  y[:20,1] = 1

  np.random.seed(1)
  np.random.shuffle(X)
  np.random.seed(1)
  np.random.shuffle(y)

  X = net2.standardize(X)
  model = net2.Net([net2.Dense(X.shape[1],10), 
                   net2.Sigmoid(),
                   net2.Dense(10,2),
                   net2.Sigmoid(),
                 ])
  model.fit(X, y, loss=net2.MSE(), optimizer=net2.SGD(0.0021, 0.011), n_epoch=100, batch_size=10)
  y_hat = model.predict(X)

  y = y.argmax(axis=1)
  y_hat = y_hat.argmax(axis=1) - 0.1
  pl.scatter(range(y.shape[0]), y, c='b', marker='>')
  pl.scatter(range(y_hat.shape[0]), y_hat, c='m', marker='o')
  pl.show()


if __name__ == '__main__':
  test_classification()
