
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
  def get_cost(self, y, y_hat):
    ''' Assumes y_hat is prob dist. '''
    a = y_hat / y_hat.sum(axis=-1, keepdims=True)
    a = np.choose(y.ravel(), a.T)
    return -np.log(a.prod()) / y.shape[0]

  def get_error(self, y, y_hat):
    z = np.empty_like(y_hat)
    a = 1 / y_hat.sum(axis=1, keepdims=True)
    z[...] = a
    b = 1 / np.choose(y.ravel(), y_hat.T)
    z[range(y.shape[0]), y.ravel()] -= b
    return z / y.shape[0]


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
