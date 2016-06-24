
import numpy as np

# loss function
############################################################
class MSE:
  def get_cost(self, y, y_hat):
    return np.mean((y - y_hat)**2, axis=0)

  def get_error(self, y, y_hat):
    return 2 * (y_hat - y)

# categorical cross entropy
class CCE:
  def normalize(self, y_hat):
    return y_hat / y_hat.sum(axis=-1, keepdims=True)

  def get_cost(self, y, y_hat):
    y_hat1 = self.normalize(y_hat)
    z = np.choose(y.ravel(), y_hat1.T)
    return -np.log(z.prod())

  def get_error(self, y, y_hat):
    a = y_hat.sum(axis=-1, keepdims=True)
    b = np.ones_like(y_hat)
    b /= a
    c = 1 / np.choose(y, y_hat.T)
    b[range(y_hat.shape[0]), y.ravel()] -= c
    return b

# optimizer
############################################################
class SGD:
  def __init__(self, lr, eta=0):
    self.lr = lr
    self.eta = eta

  def update(self, param_grad):
    for param, grad in param_grad:
      param -= (self.lr * grad + self.eta * param)


# initializer
############################################################
def glorot_normal(rng, shape):
  return rng.normal(scale=np.sqrt(2/np.sum(shape)), size=shape)


# data preprocessing
############################################################
def standardize(x):
  x -= x.mean(axis=0)
  x /= x.std(axis=0)
  return x
