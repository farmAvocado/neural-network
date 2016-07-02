
import numpy as np
import pylab as pl
import net

pl.style.use('ggplot')

def test_regression():
  X = np.random.rand(50, 2)
  y = X.sum(axis=1, keepdims=True)**2
  X = net.standardize(X)
  model = net.Net([net.Dense(X.shape[1],10), 
                   net.Relu(),
                   net.Dense(10,1),
                   net.Relu(),
                 ])
  model.fit(X, y, loss=net.MSE(), optimizer=net.SGD(0.01, 1.0), n_epoch=100, batch_size=1)
  y_hat = model.predict(X)

  pl.plot(y, 'b->')
  pl.plot(y_hat, 'm--o')
  pl.show()

def test_classification():
  X = np.random.rand(50, 2)
  X[:20,:] += 1
  y = np.zeros((50,2), dtype='int')
  y[:20,1] = 1

  np.random.seed(1)
  np.random.shuffle(X)
  np.random.seed(1)
  np.random.shuffle(y)

  X = net.standardize(X)
  model = net.Net([net.Dense(X.shape[1],10), 
                   net.Sigmoid(),
                   net.Dense(10,2),
                   net.Sigmoid(),
                 ])
  model.fit(X, y, loss=net.MSE(), optimizer=net.SGD(0.0021, 0.011), n_epoch=100, batch_size=10)
  y_hat = model.predict(X)

  y = y.argmax(axis=1)
  y_hat = y_hat.argmax(axis=1) - 0.1
  pl.scatter(range(y.shape[0]), y, c='b', marker='>')
  pl.scatter(range(y_hat.shape[0]), y_hat, c='m', marker='o')
  pl.show()

def test_classification2():
  X = np.random.rand(50, 2)
  X[:20,:] += 1
  y = np.zeros(50, dtype='int')
  y[:20] = 1

  np.random.seed(1)
  np.random.shuffle(X)
  np.random.seed(1)
  np.random.shuffle(y)

  X = net.standardize(X)
  model = net.Net([net.Dense(X.shape[1],10), 
                   net.Relu(),
                   net.Dense(10,2),
                   net.Relu(),
                 ])
  model.fit(X, y, loss=net.CCE(), optimizer=net.SGD(1e-2, 0.1), n_epoch=100, batch_size=10)
  y_hat = model.predict(X)

  y_hat = y_hat.argmax(axis=1) - 0.1
  pl.scatter(range(y.shape[0]), y, c='b', marker='>')
  pl.scatter(range(y_hat.shape[0]), y_hat, c='m', marker='o')
  pl.show()

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

  return X, y

def test_classification3():
  X, y = gen_data(5)

  np.random.seed(1)
  np.random.shuffle(X)
  np.random.seed(1)
  np.random.shuffle(y)

#  X = net.standardize(X)
  model = net.Net([net.Dense(X.shape[1],100), 
                   net.Relu(),
                   net.Dense(100,25),
                   net.Relu(),
                   net.Dense(25,5),
                   net.Sigmoid(),
                 ])
  model.fit(X, y, loss=net.CCE(), optimizer=net.SGD(1e-0, 1e-4), n_epoch=1000, batch_size=100, shuffle=True)

  z = np.meshgrid(np.arange(-1.5,1.5,0.02), np.arange(-1.5,1.5,0.02))
  X1 = np.c_[z[0].ravel(), z[1].ravel()]
  y1 = model.predict(X1)
  y1 = y1.argmax(axis=1)

  pl.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=pl.cm.Spectral)
  pl.contourf(z[0], z[1], y1.reshape(z[0].shape), cmap=pl.cm.Spectral, alpha=0.3)
  pl.show()


if __name__ == '__main__':
  test_classification3()
