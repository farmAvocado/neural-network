import numpy as np
import pylab as pl
from util import *

RNG = np.random.RandomState(17)

class Dense:
  def __init__(self, n_in, n_out):
    self.has_param = True
    self.W = RNG.normal(size=(n_out, n_in))
    self.b = RNG.normal(size=n_out)
    self.W_grad = None
    self.b_grad = None

  def forward(self, X):
    self.inp = X
    self.outp = np.dot(X, self.W.T) + self.b
    return self.outp

  def backward(self, X):
    self.W_grad = np.dot(X.T, self.inp) / X.shape[0]
    self.b_grad = X.mean(axis=0)
    self.outp = np.dot(X, self.W)
    return self.outp

  def get_param_grad(self):
    return [(self.W, self.W_grad), (self.b, self.b_grad)]

class Linear:
  def __init__(self):
    self.has_param = False

  def forward(self, X):
    self.outp = X
    return self.outp

  def backward(self, X):
    self.outp = X
    return self.outp

class Sigmoid:
  def __init__(self):
    self.has_param = False

  def forward(self, X):
    self.outp = 1.0 / (1 + np.exp(-X))
    return self.outp

  def backward(self, X):
    self.outp = self.outp * (1 - self.outp) * X
    return self.outp

class Net:
  def __init__(self, layers):
    self.layers = layers

  def fit(self, X, y, loss=MSE(), optimizer=SGD(lr=0.1, eta=0), n_epoch=1, batch_size=1):
    n = X.shape[0]
    for i in range(n_epoch):
      batches = [(X[_:_+batch_size], y[_:_+batch_size]) for _ in range(0, n, batch_size)]
      for X_batch, y_batch in batches:
        y_hat = self.forward(X_batch)
        e = loss.get_error(y_batch, y_hat)
        self.backward(e)

        for layer in self.layers:
          if layer.has_param:
            optimizer.update(layer.get_param_grad())
      print('cost:', loss.get_cost(y_batch, y_hat))

  def predict(self, X):
    return self.forward(X)

  def forward(self, X):
    for l in self.layers:
      X = l.forward(X)
    return l.outp

  def backward(self, X):
    for l in self.layers[::-1]:
      X = l.backward(X)

def grad_check():
  pass


if __name__ == '__main__':
  np.random.seed(1)
  X = np.random.rand(80, 5)
  y = (X.sum(axis=1)**3 + 1)[:,np.newaxis]

  net = Net([Dense(5,1), Linear()])
  net.fit(X, y, loss=MSE(), optimizer=SGD(lr=0.1, eta=0), n_epoch=150, batch_size=5)

  y_hat = net.predict(X)

  pl.style.use('ggplot')
  pl.plot(y, '+')
  pl.plot(y_hat, '^')
  pl.show()
