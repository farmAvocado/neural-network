
import sklearn.naive_bayes as nb
import numpy as np
from sklearn.feature_extraction import text

def load_data():
  token = '\'?[a-zA-Z0-9_]+|[,.;!?$]'
  cvec = text.CountVectorizer(
      token_pattern=token,
      ngram_range=(1,1))

  with open('../data/rt-polaritydata/rt-polarity.pos', 'r', encoding='latin1') as fp:
    sent_pos = fp.readlines()
  with open('../data/rt-polaritydata/rt-polarity.neg', 'r', encoding='latin1') as fp:
    sent_neg = fp.readlines()

  sent = sent_pos + sent_neg
  X = cvec.fit_transform(sent)
  y = np.zeros(X.shape[0], dtype='int')
  y[:len(sent_pos)] = 1

  ind = np.random.permutation(X.shape[0])
  return X[ind],y[ind],cvec

if __name__ == '__main__':
  X,y,vec = load_data()
  X_train, X_val = X[:-1000], X[-1000:]
  y_train, y_val = y[:-1000], y[-1000:]

  model = nb.BernoulliNB()
  model1 = nb.MultinomialNB()

  model.fit(X_train, y_train)
  model1.fit(X_train, y_train)
  y_hat = model.predict_log_proba(X_val) + model1.predict_log_proba(X_val)
  y_hat = y_hat.argmax(axis=1)
  print(np.mean(y_val == y_hat))
