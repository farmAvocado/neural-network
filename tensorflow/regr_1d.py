# coding: utf8

import tensorflow as tf
import numpy as np
import pylab as pl
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, Adam, Adadelta, Adagrad
from keras.regularizers import l2
from keras.callbacks import Callback

pl.ion()

def spline(x, n):
  np.random.seed(17)
  x = np.random.rand(n) * x - x/2.0
  x.sort()
  y = x * np.sin(x)
  return x,y

def take_0():
  x, y = spline(30, 20)
  l2_reg = 0

  mo = Sequential()
  mo.add(Dense(20, input_dim=1, W_regularizer=l2(l2_reg), weights=None, init='glorot_normal'))
  mo.add(Activation('relu'))
  mo.add(Dense(20, W_regularizer=l2(l2_reg), weights=None, init='glorot_normal'))
  mo.add(Activation('relu'))
  mo.add(Dense(1, W_regularizer=l2(l2_reg), weights=None, init='glorot_normal'))
  op = SGD(0.001, 0.9, 0)
  mo.compile(op, loss='mean_squared_error')

  n_iter = 1000
  n_batch = 50
  is_shuffle = False

  class Plot(Callback):
    def __init__(self, x, y):
      self.x = x
      self.y = y
      pl.scatter(x, y)
      self.y_hat, = pl.plot(self.x, np.random.rand(self.x.shape[0]), 'c-+')

    def on_train_begin(self, logs={}):
      pass

    def on_batch_end(self, batch, logs={}):
      y_hat = self.model.predict(self.x)
      self.y_hat.set_data(self.x, y_hat)
      pl.gcf().canvas.draw()

  mo.fit(x, y, batch_size=n_batch, nb_epoch=n_iter, shuffle=is_shuffle, callbacks=[Plot(x, y)])

def glorot_normal(shape, dtype=tf.float32):
  s = np.sqrt(2.0 / (shape[0] + shape[1]))
  return tf.random_normal(shape, stddev=s, dtype=dtype)

def trunc_normal(shape, stddev=0.06, dtype=tf.float32):
  return tf.truncated_normal(shape=shape, dtype=dtype, stddev=stddev)

def take_1():
  x, y = spline(20, 50)
  x.shape = (-1, 1)
  y.shape = (-1, 1)
  floatX = tf.float32
  init_w = glorot_normal

  x_val = np.linspace(-20, 20, 100)
  x_val.shape = (-1,1)

  t_x = tf.placeholder(dtype=floatX, shape=[None,1])
  t_y = tf.placeholder(dtype=floatX, shape=[None,1])

  t_w0 = tf.Variable(init_w(shape=[20,1], dtype=floatX))
  t_b0 = tf.Variable(tf.zeros(shape=[20], dtype=floatX))
  t_z0 = tf.matmul(t_x, t_w0, transpose_b=True) + t_b0
  t_a0 = tf.nn.relu(t_z0)

  t_w1 = tf.Variable(init_w(shape=[20,20], dtype=floatX))
  t_b1 = tf.Variable(tf.zeros(shape=[20], dtype=floatX))
  t_z1 = tf.matmul(t_a0, t_w1, transpose_b=True) + t_b1
  t_a1 = tf.nn.sigmoid(t_z1)

  t_w2 = tf.Variable(init_w(shape=[20,20], dtype=floatX))
  t_b2 = tf.Variable(tf.zeros(shape=[20], dtype=floatX))
  t_z2 = tf.matmul(t_a1, t_w2, transpose_b=True) + t_b2
  t_a2 = tf.nn.relu(t_z2)

  t_w3 = tf.Variable(init_w(shape=[1,20], dtype=floatX))
  t_b3 = tf.Variable(tf.zeros(shape=[1], dtype=floatX))
  t_z3 = tf.matmul(t_a2, t_w3, transpose_b=True) + t_b3
  t_a3 = t_z3

  t_y_hat = t_a3
  t_loss = tf.reduce_mean(tf.square(t_y_hat - t_y))
  t_op = tf.train.MomentumOptimizer(0.008, 0.95).minimize(t_loss)

  S = tf.Session()
  S.run(tf.initialize_all_variables())

  n_iter = 1000
  n_batch = 50
  is_shuffle = False
  ind = np.arange(x.shape[0])

  y_hat = S.run(t_y_hat, feed_dict={t_x:x_val})
  pl.scatter(x, y)
  upd, = pl.plot(x, np.random.rand(x.shape[0]), 'c-+')
  for i in range(n_iter):
    if is_shuffle:
      np.random.shuffle(ind)
    ix = 0
    total_loss = 0
    total_batch = 0
    while ix < x.shape[0]:
      _, loss = S.run([t_op, t_loss], feed_dict={
        t_x:x[ind[ix:ix+n_batch]],
        t_y:y[ind[ix:ix+n_batch]]})
      total_loss += loss
      total_batch += 1
      ix += n_batch

    print('loss: {}'.format(total_loss/total_batch))
    y_hat = S.run(t_y_hat, feed_dict={t_x:x_val})
    upd.set_data(x_val, y_hat)
    pl.gcf().canvas.draw()


if __name__ == '__main__':
#  take_0()
  take_1()
  pl.waitforbuttonpress()
