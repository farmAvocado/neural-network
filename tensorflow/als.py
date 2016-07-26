# coding: utf8

import tensorflow as tf
import numpy as np

floatX = tf.float32
intX = tf.int32
init = tf.truncated_normal

def als(data, n_u, n_v, n_f, val_data=None, n_iter=1000, batch_size=100, lr=0.01, mmt=0.9, l2=0):
  t_index = tf.placeholder(dtype=intX, shape=[None])
  t_y = tf.placeholder(dtype=floatX, shape=[None])

  # latent factors and bias for each u
  t_u = tf.Variable(init(shape=[n_u,n_f]))
  t_ub = tf.Variable(init(shape=[n_u,1]))

  # latent factors and bias for each v
  t_v = tf.Variable(init(shape=[n_f,n_v]))
  t_vb = tf.Variable(init(shape=[1,n_v]))

  t_r = tf.matmul(t_u, t_v) + t_ub + t_vb
  t_y_hat = tf.gather(tf.reshape(t_r,[-1]), t_index)
  t_loss = tf.reduce_mean(tf.square(t_y - t_y_hat)) + l2 * (tf.nn.l2_loss(t_u) + tf.nn.l2_loss(t_v))
  t_opt = tf.train.MomentumOptimizer(lr, mmt)
  alt = [t_opt.minimize(t_loss, var_list=var) for var in [[t_u,t_ub], [t_v,t_vb]]]

  S = tf.Session()
  S.run(tf.initialize_all_variables())

  index = data[:,0] * n_v + data[:,1]
  y = data[:,2]
  if val_data is not None:
    t_val_index = tf.placeholder(dtype=intX, shape=[None])
    t_val_y = tf.placeholder(dtype=floatX, shape=[None])
    t_val_y_hat = tf.gather(tf.reshape(t_r,[-1]), t_val_index)
    t_val_loss = tf.reduce_mean(tf.square(t_val_y - t_val_y_hat))
    val_index = val_data[:,0] * n_v + val_data[:,1]
    val_y = val_data[:,2]

  for i in range(n_iter):
    ix = 0
    total_loss = 0
    total_updates = 0
    while ix < index.shape[0]:
      for a in alt:
        rez = S.run([t_loss, a, t_y_hat], 
            feed_dict={t_index:index[ix:ix+batch_size], t_y:y[ix:ix+batch_size]})
        total_loss += rez[0]
        total_updates += 1
      ix += batch_size

    s = 'iter {}, train loss = {}'.format(i, total_loss/total_updates)
    if val_data is not None:
      rez = S.run([t_val_loss], feed_dict={t_val_index:val_index, t_val_y:val_y})
      s += ', val loss = {}'.format(rez[0])
    print(s)

  return S.run([t_u, t_ub, t_v, t_vb])


if __name__ == '__main__':
  import os.path
  lens = np.loadtxt(os.path.expanduser('../data/ml-100k/u.data'))
  lens[:,:2] -= 1

  u,ub,v,vb = als(lens[:80000,:3], 943, 1682, 500, lens[80000:,:3], 
                n_iter=100, batch_size=1000, lr=0.2, mmt=0.9, l2=0.001)
  r = np.dot(u, v) + ub + vb
  np.savetxt('/tmp/a', r[lens[:,0].astype('int'), lens[:,1].astype('int')], '%.2f')
