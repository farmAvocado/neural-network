
import numpy as np
import pylab as pl
import net, util

def gen_data(n_class=3):
  N = 100 # number of points per class
  D = 2 # dimensionality
  K = n_class # number of classes
  X = np.zeros((N*K,D)) # data matrix (each row = single example)
  y = np.zeros(N*K, dtype='uint8') # class labels
  for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

  return X, y[:, np.newaxis]

if __name__ == '__main__':
  X, y = gen_data(3) 
  model = net.Net([net.Dense(2,100), 
                   net.Relu(),
                   net.Dense(100,3),
                   net.Sigmoid(),
                 ])
  model.fit(X, y, loss=util.CCE(), optimizer=util.SGD(1.01, 0), n_epoch=100, batch_size=100)

  z = np.meshgrid(np.arange(-1,1,0.02), np.arange(-1,1,0.02))
  X1 = np.c_[z[0].ravel(), z[1].ravel()]
  y1 = model.predict(X1)
  y1 = y1.argmax(axis=1)

  fig = pl.figure()
  pl.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=pl.cm.Spectral)
  pl.contourf(z[0], z[1], y1.reshape(z[0].shape), cmap=pl.cm.Spectral, alpha=0.5)
  pl.show()
