
import numpy as np

# loss function
############################################################
class MSE:
  def get_cost(self, y, y_hat):
    return 0.5 * np.mean((y - y_hat)**2, axis=0)

  def get_error(self, y, y_hat):
    return y_hat - y

class CE:
  def get_cost(self, y, y_hat):
    pass


# optimizer
############################################################
class SGD:
  def __init__(self, lr, eta=0):
    self.lr = lr
    self.eta = eta

  def update(self, param_grad):
    for param, grad in param_grad:
      param -= (self.lr * grad + self.eta * param)
