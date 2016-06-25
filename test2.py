
import numpy as np
import pylab as pl
import net2

pl.style.use('ggplot')

def gen_data():
  X = np.random.rand(50, 2)
  y = X.sum(axis=1, keepdims=True)**2
  return X, np.sin(y)

if __name__ == '__main__':
  X, y = gen_data()
  X = net2.standardize(X)
  model = net2.Net([net2.Dense(X.shape[1],10), 
                   net2.Relu(),
                   net2.Dense(10,1),
                   net2.Relu(),
                 ])
  model.fit(X, y, loss=net2.MSE(), optimizer=net2.SGD(0.01, 1.0), n_epoch=100, batch_size=1)
  y_hat = model.predict(X)

  pl.plot(y, 'b->')
  pl.plot(y_hat, 'm--o')
  pl.show()
