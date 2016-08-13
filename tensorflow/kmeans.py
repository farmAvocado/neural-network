
import tensorflow as tf
import numpy as np
import pylab as pl

floatX = tf.float64
intX = tf.int64


def km(x, n_cluster=3, n_iter=100):
  n_data = x.shape[0]
  n_feat = x.shape[1]

  input_x = tf.placeholder(shape=[n_data,n_feat], dtype=floatX)
  label = tf.Variable(np.arange(n_data, dtype=np.int64)%n_cluster)

  n_label = tf.cast(tf.reduce_max(label) + 1, tf.int32)
  sums = tf.unsorted_segment_sum(input_x, label, n_label)
  counts = tf.unsorted_segment_sum(tf.ones_like(label, dtype=floatX), label, n_label)
  means = sums / tf.reshape(counts, [-1,1])

  dist = tf.reshape(input_x, [n_data,1,n_feat]) - \
      tf.reshape(means, [1,n_label,n_feat])
  dist = tf.reduce_sum(tf.square(dist), 2)

  updates = [label.assign(tf.argmin(dist, 1))]

  S = tf.Session()
  S.run(tf.initialize_all_variables())
  for i in range(n_iter):
    rez = S.run(updates, feed_dict={input_x:x})

  return rez[0]

if __name__ == '__main__':
  x = np.random.rand(100,2) 
  x[:20] += 1.0
  x[-20:] -= 1.0
  y = km(x, 5)
  pl.scatter(x[:,0], x[:,1], c=y)
  pl.show()
