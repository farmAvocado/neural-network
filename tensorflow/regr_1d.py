# coding: utf8

from net import *
import numpy as np
import pylab as pl


def spline(x, n):
  np.random.seed(17)
  x = np.random.rand(n) * x - x/2.0
  x.sort()
  y = x * np.sin(x)
  return x,y

class Plot1D:
  def __init__(self, x):
    self.x = x
    self.upd, = pl.plot(x, np.random.rand(x.shape[0]), 'c-+')

  def __call__(self, model):
    y = model.predict(self.x)
    self.upd.set_data(self.x, y)
    pl.gcf().canvas.draw()

if __name__ == '__main__':
  model = Net([Dense(1,10), Relu(), Dense(10, 10), Sigmoid(), Dense(10,1)])
  model.build('regression')

  x, y = spline(20, 50)
  x.shape = (-1,1)
  y.shape = (-1,1)
  data_iter = DataIter(x, y)

  pl.ion()
  pl.scatter(x, y)
  model.fit(data_iter, 3000, 50, lr=0.01, dstep=500, 
      callbacks=[Plot1D(np.linspace(-10, 10, 100).reshape(-1,1))])
  pl.waitforbuttonpress()
