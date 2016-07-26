# coding: utf8

import tensorflow as tf
import numpy as np


if __name__ == '__main__':
  floatX = tf.float32
  intX = tf.int32
  init = tf.truncated_normal

  lens = np.loadtxt('../data/ml-100k/u.data')

  n_u = 944
  n_v = 1683
  n_f = 100

  u_ind = lens[:,0]
  v_ind = lens[:,1]
  y = lens[:,2]

  t_u_ind = tf.placeholder(dtype=intX, shape=[None])
  t_v_ind = tf.placeholder(dtype=intX, shape=[None])
  t_ind = t_u_ind * n_v + t_v_ind
  t_y = tf.placeholder(dtype=floatX, shape=[None])

  t_u = tf.Variable(init(shape=[n_u, n_f]))
  t_v = tf.Variable(init(shape=[n_v, n_f]))
  t_ub = tf.Variable(init(shape=[n_u,1]))
  t_vb = tf.Variable(init(shape=[1, n_v]))
  t_r = tf.matmul(t_u, t_v, transpose_b=True) + t_ub + t_vb

  t_y_hat = tf.gather(tf.reshape(t_r, [-1]), t_ind)

  t_loss = tf.reduce_mean(tf.square(t_y - t_y_hat)) + 0.1 * (tf.reduce_sum(tf.square(t_u)) + tf.reduce_sum(tf.square(t_v)))
  t_opt = tf.train.MomentumOptimizer(0.1, 0.9)
  t_train = t_opt.minimize(t_loss)

  S = tf.Session()
  S.run(tf.initialize_all_variables())

  for i in range(2000):
    rez = S.run([t_loss, t_train, t_r], feed_dict={t_u_ind:u_ind, t_v_ind:v_ind, t_y:y})
    print(rez[0]) #, rez[2], y)

  np.savetxt('/tmp/a', rez[2], '%.3f')
