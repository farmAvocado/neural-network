# coding: utf8

import tensorflow as tf
import numpy as np

floatX = tf.float32
intX = tf.int32

def glorot_normal(shape, dtype=floatX):
  s = np.sqrt(2.0 / np.sum(shape))
  return tf.random_normal(shape=shape, dtype=dtype, stddev=s)


class Dense:
  def __init__(self, n_in, n_out, l2_reg=0, init_w=glorot_normal, init_b=tf.zeros):
    self.n_in = n_in
    self.n_out = n_out
    self.t_w = tf.Variable(init_w(shape=[n_out,n_in]))
    self.t_b = tf.Variable(init_b(shape=[n_out]))
    self.t_reg = l2_reg * tf.nn.l2_loss(self.t_w)

  def build(self, t_x):
    self.t_z = tf.matmul(t_x, self.t_w, transpose_b=True) + self.t_b


class Relu:
  def build(self, t_x):
    self.t_z = tf.nn.relu(t_x)


class Sigmoid:
  def build(self, t_x):
    self.t_z = tf.nn.sigmoid(t_x)


class Net:
  def __init__(self, layers=[]):
    self.layers = layers

  def build(self, task):
    self.task = task
    n_layer = len(self.layers)
    n_in = 0
    n_out = 0

    for i in range(n_layer):
      if n_in == 0:
        if hasattr(self.layers[i], 'n_in'):
          n_in = self.layers[i].n_in
      if hasattr(self.layers[i], 'n_out'):
        n_out = self.layers[i].n_out

    self.t_x = tf.placeholder(dtype=floatX, shape=[None,n_in])
    if task == 'regression':
      self.t_y = tf.placeholder(dtype=floatX, shape=[None,n_out])
    elif task == 'classification':
      # n_out should be 1, assume sparse form
      self.t_y = tf.placeholder(dtype=intX, shape=[None,n_out])

    t_out = self.t_x
    t_reg = 0
    for i in range(n_layer):
      layer = self.layers[i]
      layer.build(t_out)
      if hasattr(layer, 't_reg'):
        t_reg += layer.t_reg
      t_out = layer.t_z

    self.t_y_hat = t_out
    if task == 'regression':
      self.t_loss = tf.reduce_mean(tf.square(t_out - self.t_y)) + t_reg
    elif task == 'classification':
      self.t_loss = tf.reduce_mean(tf.sparse_softmax_cross_entropy_with_logits(t_out, self.t_y)) + t_reg

  def fit(self, data_iter, n_iter, batch_size, lr=0.01, momentum=0, dr=1, dstep=1000, every=1, shuffle=False, callbacks=[]):
    t_gstep = tf.Variable(0, trainable=False)
    t_lr = tf.train.exponential_decay(lr, t_gstep, dstep, dr, staircase=True)
    self.t_opt = tf.train.MomentumOptimizer(t_lr, momentum).minimize(self.t_loss)

    self.S = tf.Session()
    self.S.run(tf.initialize_all_variables())

    total_loss = 0
    total_updates = 0
    for i in range(n_iter):
      for x_batch,y_batch in data_iter(batch_size, shuffle):
        _, loss = self.S.run([self.t_opt, self.t_loss],
            feed_dict={self.t_x: x_batch, self.t_y: y_batch})
        total_loss += loss
        total_updates += 1

      if i % every == 0:
        print('iter = {}, loss = {}'.format(i, total_loss/total_updates))
        total_loss = 0
        total_updates = 0

        for cb in callbacks:
          cb(self)

  def predict(self, x):
    y_hat = self.S.run(self.t_y_hat, feed_dict={self.t_x: x})
    if self.task == 'classification':
      return np.argmax(y_hat, axis=1)
    return y_hat


class DataIter:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.n = x.shape[0]
    self.ind = np.arange(self.n)

  def __call__(self, batch_size, shuffle=False):
    self.batch_size = batch_size
    if shuffle:
      np.random.shuffle(self.ind)
    return iter(self)

  def __iter__(self):
    ix = 0
    while ix < self.n:
      ix_batch = self.ind[ix:ix+self.batch_size]
      yield self.x[ix_batch], self.y[ix_batch]
      ix += self.batch_size


