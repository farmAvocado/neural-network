
import numpy as np
import cynet

def test_dense():
  x = np.arange(6, dtype='double').reshape(2,3)
  z = np.random.rand(2,2).astype('double')
  h = 1e-5

  fc = cynet.Dense(3, 2)
  fc.W = np.arange(6, dtype='double').reshape(2,3)

  W_grad = np.zeros_like(fc.W)
  it = np.nditer(fc.W, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    ix = it.multi_index
    oldv = fc.W[ix]

    fc.W[ix] = oldv + h
    fc.forward(x)
    y0 = np.sum(fc.outp * z)
    fc.W[ix] = oldv - h
    fc.forward(x)
    y1 = np.sum(fc.outp * z)
    x[ix] = oldv

    W_grad[ix] = (y0 - y1) / (h + h)
    it.iternext()

  fc.W = np.arange(6, dtype='double').reshape(2,3)
  fc.backward(z)

  print(np.allclose(W_grad, fc.W_grad))


if __name__ == '__main__':
  test_dense()
