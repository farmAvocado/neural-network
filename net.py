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

  def train(self, X, y, loss=MSE(), optimizer=SGD(lr=0.1, eta=0), n_epoch=1, batch_size=1):
    n = X.shape[0]
    for i in range(n_epoch):
      batches = [(X[_:_+batch_size], y[_:_+batch_size]) for _ in range(0, n, batch_size)]
      for X_batch, y_batch in batches:
        y_hat = self.forward(X_batch)
        e = loss.get_error(y_batch, y_hat, 1)
        self.backward(e)

        for layer in self.layers:
          if layer.has_param:
            optimizer.update(layer.get_param_grad())
      print('cost:', loss.get_cost(y_batch, y_hat, 1))

  def train1(self, X, y, lr=0.005, c=0, batch_size=100, n_epochs=100):
    for it in range(n_epochs):
      batch = [(X[_:_+batch_size], y[_:_+batch_size]) for _ in range(0, X.shape[0], batch_size)]
      for X_batch, y_batch in batch:
        y_hat = self.forward(X_batch)
        error = (y_hat - y_batch)
        self.backward(error)

        for l in self.layers:
          if l.has_param:
            l.W = l.W - lr * l.W_grad - c * l.W
            l.b = l.b - lr * l.b_grad
#      print('error:', np.sum(error**2))

  def forward(self, X):
    for l in self.layers:
      X = l.forward(X)
    return l.outp

  def backward(self, X):
    for l in self.layers[::-1]:
      X = l.backward(X)

def grad_check():
  X = np.arange(17).reshape(1, 17)
  y = np.array([[2]])

  net = Net([Dense(17,1), Linear()])
  y_hat = net.forward(X)
  e = y_hat - y
  net.backward(e)

  e = 0.5 * (y_hat - y)**2
  W_grad = net.layers[0].W_grad.copy()
  b_grad = net.layers[0].b_grad
  W_grad1 = np.empty_like(net.layers[0].W_grad)
  b_grad1 = 0
  W = net.layers[0].W
  for i in range(W.shape[0]):
    for j in range(W.shape[1]):
      W[i,j] += 1e-5
      e1 = 0.5 * (net.forward(X) - y)**2
      W_grad1[i,j] = (e1 - e) / 1e-5
      W[i,j] -= 1e-5

  net.layers[0].b += 1e-5
  e1 = 0.5 * (net.forward(X) - y)**2
  b_grad1 = (e1 - e) / 1e-5


  print(b_grad)
  print(b_grad1)


if __name__ == '__main__':
  np.random.seed(1)
  X = np.random.rand(20, 5)
  y = np.sin(X.sum(axis=1))[:,np.newaxis]

  net = Net([Dense(5,5), Sigmoid(), Dense(5,1), Sigmoid()])
#  net.train(X, y, n_epochs=1, lr=1.5, batch_size=10)
  net.train1(X, y, loss=MSE(), optimizer=SGD(lr=1.5, eta=0), n_epoch=500, batch_size=10)

  y_hat = net.forward(X)

  pl.style.use('ggplot')
  pl.plot(y, '+')
  pl.plot(y_hat, '^')
  pl.show()
