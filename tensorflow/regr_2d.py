# coding: utf8

from net import *
import numpy as np
import pylab as pl

pl.ion()

def image(sample=1):
  from skimage.io import imread
  img = imread('../data/cat.jpg')
  return img[::sample,::sample,:3]


class PixelIter:
  def __init__(self, img):
    self.w = img.shape[0]
    self.h = img.shape[1]
    self.img = img

  def __call__(self, batch_size, shuffle=True):
    self.batch_size = batch_size
    return iter(self)

  def __iter__(self):
    x = np.random.randint(0, self.w, size=self.batch_size)
    y = np.random.randint(0, self.h, size=self.batch_size)

    pix = self.img[x,y] / 255.0
    pos = np.vstack(((x-self.w/2)/self.w, (y-self.h/2)/self.h)).T
    yield pos, pix


class PixelPlot:
  def __init__(self, w, h):
    self.w = w
    self.h = h
    x = np.arange(w)
    y = np.arange(h)
    y,x = np.meshgrid(x,y)
    x.shape = (-1,)
    y.shape = (-1,)
    self.pos = np.vstack(((x-w/2)/w, (y-h/2)/h)).T
    self.upd = pl.imshow(np.random.rand(w,h))

  def __call__(self, model):
    pix = model.predict(self.pos)
#    pix = np.floor(pix*255)
    pix.shape = (self.w, self.h, 3)
    self.upd.set_data(pix)
    pl.gcf().canvas.draw()


if __name__ == '__main__':
  img = image()
  piter = PixelIter(img)

  model = Net([Dense(2, 20), Relu(),
              Dense(20, 20), Relu(),
              Dense(20, 20), Relu(),
              Dense(20, 20), Relu(),
              Dense(20, 20), Relu(),
              Dense(20, 20), Relu(),
              Dense(20, 20), Relu(),
              Dense(20, 3)
              ])
  model.build('regression')

  model.fit(piter, 20000, 500, every=100, callbacks=[PixelPlot(img.shape[0], img.shape[1])])
  pl.waitforbuttonpress()
