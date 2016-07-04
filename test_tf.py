# coding: utf8

import tensorflow as tf
import numpy as np
import pylab as pl

def spiral_data(n, k):
  p = np.pi * 2 / k
  r = np.arange(0, 1, 1.0/n)
  t = np.arange(0, p, p/n)

  x = []
  y = []
  for i in range(k):
    z = t + i*p
    x += map(lambda _r,_z: (_r*np.cos(_z), _r*np.sin(_z)), r, z)
    y += [i] * len(z)

  x = np.array(x)
  y = np.array(y)
  return x,y

def softmax(inp_x, inp_y):
  x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
  y_ = tf.placeholder(dtype=tf.int32, shape=[None])

  w = tf.Variable(tf.truncated_normal(shape=[100, 2], dtype=tf.float32))
  b = tf.Variable(tf.zeros(shape=[100], dtype=tf.float32))
  z = tf.matmul(x, w, transpose_b=True) + b
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(z, y_) + tf.nn.l2_loss(w))

  opt = tf.train.GradientDescentOptimizer(0.1)
  train_op = opt.minimize(loss)
  init_op = tf.initialize_all_variables()

  with tf.Session() as sess:
    sess.run(init_op)
    for i in range(10):
      _, loss_val = sess.run([train_op, loss], feed_dict={x:inp_x, y_:inp_y})
      print(loss_val)


if __name__ == '__main__':
  x,y = spiral_data(10, 3)
#  pl.scatter(x[:,0], x[:,1], c=y, s=40, marker='o', cmap=pl.cm.Spectral)
#  pl.show()
  softmax(x, y)

