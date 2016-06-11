
import numpy as np

# loss function
############################################################
class MSE:
  def get_cost(self, y, y_hat, n):
    return 0.5 * np.sum((y - y_hat)**2) / n

  def get_error(self, y, y_hat, n):
    return (y_hat - y) / n


# optimizer
############################################################
class SGD:
  def __init__(self, lr, eta=0):
    self.lr = lr
    self.eta = eta

  def update(self, param_grad):
    for param, grad in param_grad:
      param -= (self.lr * grad + self.eta * param)
