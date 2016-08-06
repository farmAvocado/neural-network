
import tensorflow as tf
import numpy as np
from scipy import sparse

floatX = tf.float64
intX = tf.int64

def solve(A, y):
  t_A = tf.placeholder(dtype=floatX, shape=A.shape)
  t_y = tf.placeholder(dtype=floatX, shape=y.shape)
  t_x = tf.matrix_solve_ls(t_A, t_y, l2_regularizer=1e-3)

  S = tf.Session()
  S.run(tf.initialize_all_variables())

  rez = S.run(t_x, feed_dict={t_A:A, t_y:y})
  return rez

def test():
  A = np.arange(16).reshape(4,4)
  y = np.arange(4).reshape(4,1)
  x = solve(A, y)
  print(x)
  print(np.dot(A, x))


class SGD:
  def __init__(self, n_u, n_v, n_latent, lr=1e-2, mmt=0.9, l2=0):
    self.input_u_index = tf.placeholder(dtype=intX, shape=[None])
    self.input_v_index = tf.placeholder(dtype=intX, shape=[None])
    self.input_pref = tf.placeholder(dtype=floatX, shape=[None])
    self.input_conf = tf.placeholder(dtype=floatX, shape=[None])

    U = tf.Variable(tf.random_uniform(dtype=floatX, shape=[n_u,n_latent], minval=0, maxval=1), name='U')
    V = tf.Variable(tf.random_uniform(dtype=floatX, shape=[n_v,n_latent], minval=0, maxval=1), name='V')
    U_embed = tf.nn.embedding_lookup(U, self.input_u_index)
    V_embed = tf.nn.embedding_lookup(V, self.input_v_index)
    Ub = tf.Variable(tf.constant(0.1, shape=[n_u], dtype=floatX))
    Vb = tf.Variable(tf.constant(0.1, shape=[n_v], dtype=floatX))

    R = tf.matmul(U_embed, V_embed, transpose_b=True)
    self.y_hat = tf.diag_part(R) + tf.gather(Ub, self.input_u_index) \
                  + tf.gather(Vb, self.input_v_index)

    self.loss = tf.reduce_mean(self.input_conf * tf.square(self.input_pref - self.y_hat))
    self.loss_with_reg = self.loss + l2 * (tf.nn.l2_loss(U) + tf.nn.l2_loss(V))

    self.opt = tf.train.MomentumOptimizer(lr, mmt)
    #self.opt = tf.train.AdamOptimizer()
    self.train = self.opt.minimize(self.loss_with_reg)

    self.S = tf.Session()

  def fit(self, data_iter, log_every=100, val_data_iter=None):
    # initialize or reset all variables' value
    self.S.run(tf.initialize_all_variables())

    total_loss = 0
    total_batches = 0
    # data_iter should yield batches, as a dict
    for batch in data_iter:
      u_index = batch['u_index']
      v_index = batch['v_index']
      pref = batch['pref']
      conf = batch['conf']

      rez = self.S.run([self.train, self.loss], feed_dict={
                        self.input_u_index: u_index,
                        self.input_v_index: v_index,
                        self.input_pref: pref,
                        self.input_conf: conf
                      })
      total_loss += rez[1]
      total_batches += 1

      if total_batches % log_every == 0:
        train_loss = total_loss / total_batches

        val_loss = None
        if val_data_iter:
          total_val_loss = 0
          total_val_batches = 0
          for batch in val_data_iter:
            u_index = batch['u_index']
            v_index = batch['v_index']
            pref = batch['pref']
            conf = batch['conf']

            rez = self.S.run([self.loss], feed_dict={
                              self.input_u_index: u_index,
                              self.input_v_index: v_index,
                              self.input_pref: pref,
                              self.input_conf: conf
                            })
            total_val_loss += rez[0]
            total_val_batches += 1
          val_loss = total_val_loss / total_val_batches

        print('{} updates, train_loss = {}, val_loss = {}'.format(total_batches, train_loss, val_loss))

class MLImplicitData:
  def __init__(self, fname, train_split=0.8):
    raw = np.loadtxt(fname)
    u_index = raw[:,0].astype('int') - 1
    v_index = raw[:,1].astype('int') - 1
    conf = raw[:,2].astype('float')
    pref = (conf > 0).astype('float')

    self.n_u = u_index.max() + 1
    self.n_v = v_index.max() + 1

    conf = sparse.csr_matrix((conf, (u_index,v_index)), shape=[self.n_u,self.n_v])
    pref = sparse.csr_matrix((pref, (u_index,v_index)), shape=[self.n_u,self.n_v])

    grid = np.meshgrid(np.arange(self.n_u), np.arange(self.n_v), indexing='ij')
    self.u_index = grid[0].ravel()
    self.v_index = grid[1].ravel()
    self.conf = conf.toarray().ravel()
    self.pref = pref.toarray().ravel()

    index = np.random.permutation(self.n_u * self.n_v)
    n_train = int(raw.shape[0] * train_split)
    self.train_index = index[:n_train]
    self.val_index = index[n_train:]

  def __call__(self, flag='train', shuffle=False, n_iter=1, batch_size=100):
    if flag == 'train':
      return DictBatchIter(
            self.u_index[self.train_index],
            self.v_index[self.train_index],
            self.pref[self.train_index],
            self.conf[self.train_index],
            shuffle,
            n_iter,
            batch_size)
    else:
      return DictBatchIter(
            self.u_index[self.val_index],
            self.v_index[self.val_index],
            self.pref[self.val_index],
            self.conf[self.val_index],
            shuffle,
            n_iter,
            batch_size)

class DictBatchIter:
  def __init__(self, u_index, v_index, pref, conf, shuffle=True, n_iter=1, batch_size=100):
    self.u_index = u_index
    self.v_index = v_index
    self.pref = pref
    self.conf = conf

    self.shuffle = shuffle
    self.n_iter = n_iter
    self.batch_size = batch_size
    self._index = np.arange(self.u_index.shape[0])

  def __iter__(self):
    for it in range(self.n_iter):
      if self.shuffle:
        np.random.shuffle(self._index)
      ix = 0
      while ix < self._index.shape[0]:
        index = self._index[ix:ix+self.batch_size]
        batch = {
            'u_index': self.u_index[index],
            'v_index': self.v_index[index],
            'pref': self.pref[index],
            'conf': self.conf[index]
            }
        yield batch
        ix += self.batch_size


if __name__ == '__main__':
  data_iter = MLImplicitData('../data/ml-100k/u.data', train_split=0.8)

  mf_sgd = SGD(n_u=data_iter.n_u, n_v=data_iter.n_v, n_latent=100, lr=0.1, l2=1e-2)
  mf_sgd.fit(data_iter('train', n_iter=100, batch_size=1000, shuffle=True), log_every=1000,
      val_data_iter=data_iter('val', n_iter=1, batch_size=1000, shuffle=True))
