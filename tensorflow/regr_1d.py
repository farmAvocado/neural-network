# coding: utf8

from net import *
import numpy as np
import pylab as pl


def spline(x, n):
  np.random.seed(17)
  x = np.random.rand(n) * x - x/2.0
  y = x * np.sin(x)
  return x,y

def slope(x, n):
  np.random.seed(17)
  x = np.random.rand(n) * x  - x/2.0
  y = x
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
  model = Net([Dense(1,20), Relu(),
              Dense(20,20), Sigmoid(),
              Dense(20,1)])
  model.build('regression')

  x, y = spline(10, 20)
  x.shape = (-1,1)
  y.shape = (-1,1)
  data_iter = DataIter(x, y)

  pl.ion()
  pl.scatter(x, y)
  model.fit(data_iter, 500, 1, every=1, lr=0.01, momentum=0, dstep=500000, shuffle=False,
      callbacks=[Plot1D(np.linspace(x.min(), x.max(), 50).reshape(-1,1))])
  pl.waitforbuttonpress()
