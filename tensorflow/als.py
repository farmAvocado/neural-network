# coding: utf8

import tensorflow as tf
import numpy as np


if __name__ == '__main__':
  floatX = tf.float32
  intX = tf.int32
  init = tf.truncated_normal

  n_u = 100
  n_v = 100
  n_f = 100

  t_u_ind = tf.placeholder(dtype=intX, shape=[None,1])
  t_v_ind = tf.placeholder(dtype=intX, shape=[None,1])
  t_uv_ind = tf.concat(1, [t_v_ind, t_v_ind])
  t_y = tf.placeholder(dtype=floatX, shape=[None])

  t_u = tf.Variable(init(shape=[n_u, n_f]))
  t_v = tf.Variable(init(shape=[n_v, n_f]))
  t_r = tf.matmul(t_u, t_v, transpose_b=True)
  t_y_hat = tf.gather_nd(t_r, t_uv_ind)

  t_loss = tf.nn.l2_loss(t_y - t_y_hat)
  t_opt = tf.train.MomentumOptimizer(0.01, 0.9)
  t_train = t_opt.minimize(t_loss)

  S = tf.Session()
  S.run(tf.initialize_all_variables())

  for i in range(100):
    rez = S.run([t_loss, t_train], feed_dict={t_u:, t_v:})
