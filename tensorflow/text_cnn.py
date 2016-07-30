# coding: utf8

import tensorflow as tf
import numpy as np

intX = tf.int64
floatX = tf.float32

class CNN:
  def __init__(self, n_words, n_classes, vocab_size,
      n_latent, filter_sizes, n_filter, l2=0.0):
    self.t_x = tf.placeholder(dtype=intX, shape=[None,n_words], name='input_x')
    self.t_y = tf.placeholder(dtype=intX, shape=[None], name='input_y')
    self.t_dropout_keep = tf.placeholder(dtype=floatX, name='dropout_keep')

    t_l2_loss = 0

    with tf.name_scope('embedding'):
      t_w = tf.Variable(tf.random_uniform([vocab_size,n_latent],-1,1), name='W')
      t_embeded = tf.nn.embedding_lookup(t_w, self.t_x)
      t_embeded = tf.expand_dims(t_embeded, -1)

    pools = []
    for f in filter_sizes:
      size = f
      with tf.name_scope('conv_{}'.format(size)):
        shape = [size,n_latent,1,n_filter]
        t_w = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='W')
        t_b = tf.Variable(tf.constant(0.1, shape=[n_filter]), name='b')
        t_conved = tf.nn.conv2d(t_embeded, t_w, strides=[1,1,1,1], padding='VALID', name='conv')
        # bias_add, t_b a 1-D tensor with the size matching the last dim of t_conved
        t_a = tf.nn.relu(tf.nn.bias_add(t_conved, t_b), name='relu')
        t_pool = tf.nn.max_pool(t_a, [1,n_words-size+1,1,1],
                              strides=[1,1,1,1], padding='VALID', name='pool')
        pools.append(t_pool)

    # concat all features learnt in filters into a long feature vector
    n_feat = n_filter * len(filter_sizes)
    t_feat = tf.concat(3, pools)
    t_feat = tf.reshape(t_feat, [-1, n_feat], name='feature_vector')
  
    # dropout
    with tf.name_scope('dropout'):
      t_drop = tf.nn.dropout(t_feat, self.t_dropout_keep, name='dropout')
  
    # output logits
    with tf.name_scope('output'):
      #t_w = tf.Variable(tf.truncated_normal([n_feat,n_classes], stddev=0.1), name='W')
      t_w = tf.get_variable('W', shape=[n_feat,n_classes],
                initializer=tf.contrib.layers.xavier_initializer())
      t_b = tf.Variable(tf.constant(0.1, shape=[n_classes]), name='b')
      t_l2_loss += tf.nn.l2_loss(t_w)
      t_l2_loss += tf.nn.l2_loss(t_b)
      t_logits = tf.nn.xw_plus_b(t_drop, t_w, t_b, name='logits')
      self.t_pred = tf.argmax(t_logits, 1, name='prediction')
  
    # loss
    with tf.name_scope('loss'):
      t_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(t_logits, self.t_y), name='loss')
      self.t_loss = t_loss + l2 * t_l2_loss
      self.t_acc = tf.reduce_mean(tf.cast(tf.equal(self.t_pred, self.t_y), floatX), name='acc')

def tokenize_file(fname, token):
  toks = []
  with open(fname, 'r', encoding='latin1') as fp:
    for l in fp:
      toks.append(token.findall(l))
  return toks

def vocab_transform(toks):
  vocab = {}
  max_length = 0
  for tok in toks:
    n = len(tok)
    if n > max_length:
      max_length = n
    for t in tok:
      if t not in vocab:
        vocab[t] = len(vocab) + 1

  n = len(toks)
  X = np.zeros((n,max_length), dtype='int')
  for i in range(n):
    for j in range(len(toks[i])):
      X[i,j] = vocab[toks[i][j]]
  return X, vocab

def load_data():
  import re
  token = re.compile('\'?[a-zA-Z0-9_]+|[,.;!?$]')
  toks_pos = tokenize_file('../data/rt-polaritydata/rt-polarity.pos', token)
  toks_neg = tokenize_file('../data/rt-polaritydata/rt-polarity.neg', token)
  toks = toks_pos + toks_neg

  X, vocab = vocab_transform(toks)
  y = np.zeros(len(toks), dtype='int')
  y[:len(toks_pos)] = 1

  ind = np.random.permutation(X.shape[0])
  return X[ind], y[ind], vocab

def data_iter(X, y, n_iter=10, batch_size=100, shuffle=True):
  for i in range(n_iter):
    ind = np.arange(X.shape[0])
    if shuffle:
      np.random.shuffle(ind)
    ix = 0
    while ix < X.shape[0]:
      X_batch = X[ind[ix:ix+batch_size]]
      y_batch = y[ind[ix:ix+batch_size]]
      ix += batch_size
      yield X_batch,y_batch


if __name__ == '__main__':
  X,y,vocab = load_data()
  model = CNN(X.shape[1], 2, len(vocab) + 1, 128, [3,4,5], 128, 0.001)

  # train
  t_opt = tf.train.AdamOptimizer(0.01)
  #t_opt = tf.train.MomentumOptimizer(0.01, 0.9)
  t_gstep = tf.Variable(0, dtype=intX, trainable=False)
  t_train = t_opt.minimize(model.t_loss, global_step=t_gstep)

  S = tf.Session()
  S.run(tf.initialize_all_variables())

  def training(X, y):
    fd = {
          model.t_x: X,
          model.t_y: y,
          model.t_dropout_keep: 0.5
        }
    _, gstep, loss, acc = S.run([t_train, t_gstep, model.t_loss, model.t_acc], 
                                feed_dict=fd)
    print('step {}, loss = {}, acc = {}'.format(gstep, loss, acc))

  def validating(X, y):
    fd = {
          model.t_x: X,
          model.t_y: y,
          model.t_dropout_keep: 1.0
        }
    gstep, loss, acc = S.run([t_gstep, model.t_loss, model.t_acc], 
                                feed_dict=fd)
    print('validating:')
    print('step {}, loss = {}, acc = {}'.format(gstep, loss, acc))
    print('')

  X_train, X_val = X[:-1000], X[-1000:]
  y_train, y_val = y[:-1000], y[-1000:]
  for X_batch, y_batch in data_iter(X_train, y_train, batch_size=100, n_iter=200):
    gstep = tf.train.global_step(S, t_gstep)
    if gstep % 100 == 0:
      validating(X_val, y_val)

    training(X_batch, y_batch)
