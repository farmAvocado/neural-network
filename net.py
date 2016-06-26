
import numpy as np

# connection layer
############################################################
class Dense:
  def __init__(self, n_in, n_out):
    self.has_W = True
    self.n_in = n_in
    self.n_out = n_out
    self.W = np.random.rand(n_out, n_in)
    self.b = np.random.rand(n_out)

  def forward(self, X):
    self.inp = X
    self.outp = X.dot(self.W.T) + self.b
    return self.outp

  def backward(self, X):
    self.W_g = np.dot(X.T, self.inp)
    self.b_g = X.sum(axis=0)
    a = X.dot(self.W)
    return a

  def get_param_grad(self):
    return [(self.W, self.W_g), (self.b, self.b_g)]

# activation layer
############################################################
class Relu:
  def __init__(self):
    self.has_W = False

  def forward(self, X):
    self.outp = np.maximum(0, X)
    return self.outp

  def backward(self, X):
    a = (self.outp > 0).astype('double')
    a *= X
    return a

class Sigmoid:
  def __init__(self):
    self.has_W = False

  def forward(self, X):
    self.outp = 1 / (1 + np.exp(-X))
    return self.outp

  def backward(self, X):
    a = self.outp * (1 - self.outp)
    a *= X
    return a


# optimizer
############################################################
class SGD:
  def __init__(self, lr, l2=0):
    self.lr = lr
    self.l2 = l2

  def update(self, param_grad):
    for param, grad in param_grad:
      param -= self.lr * (grad + self.l2 * param)


# loss function
############################################################
class MSE:
  def get_cost(self, y, y_hat):
    return np.mean((y - y_hat)**2, axis=0)

  def get_error(self, y, y_hat):
    return 2 * (y_hat - y) / y.shape[0]

class CCE:
  def get_cost(self, y, y_hat):
    a = y_hat / y_hat.sum(axis=1, keepdims=True)
    b = y.choose(a.T)
    return -np.log(b.prod()) / y.shape[0]

  def get_error(self, y, y_hat):
    a = 1 / y_hat.sum(axis=1, keepdims=True)
    b = 1 / y.choose(y_hat.T)
    z = a.repeat(y_hat.shape[1], axis=1)
    z[range(y.shape[0]),y] -= b
    z /= y.shape[0]
    return z


# net
############################################################
class Net:
  def __init__(self, layers):
    self.layers = layers

  def fit(self, X, y, loss=MSE(), optimizer=SGD(lr=0.1, l2=0), n_epoch=1, batch_size=1, shuffle=True):
    n = X.shape[0]
    for i in range(n_epoch):
#      if shuffle:
#        seed = np.random.randint(n_epoch)
#        np.random.seed(seed)
#        np.random.shuffle(X)
#        np.random.seed(seed)
#        np.random.shuffle(y)

      batches = [(X[_:_+batch_size], y[_:_+batch_size]) for _ in range(0, n, batch_size)]
      for X_batch, y_batch in batches:
        y_hat = self.forward(X_batch)
        e = loss.get_error(y_batch, y_hat)
        self.backward(e)

        for layer in self.layers:
          if layer.has_W:
            optimizer.update(layer.get_param_grad())
      print('cost:', loss.get_cost(y_batch, y_hat))

  def predict(self, X):
    return self.forward(X)

  def forward(self, X):
    for l in self.layers:
      X = l.forward(X)
    return l.outp

  def backward(self, X):
    for l in self.layers[::-1]:
      X = l.backward(X)


# util
############################################################
def standardize(X):
  mean = X.mean(axis=0)
  X -= mean
  std = X.std(axis=0)
  X /= std
  return X
