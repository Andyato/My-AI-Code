import numpy as np 
import utils.model_selection as ms

class LogisticRegressionClassifier(object):
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        self._w = None

    def _sigmoid(self, z):
        return 1. / (1. + np.exp(-z))

    def _J(self, w, X, y):
        y_hat = self._sigmoid(X.dot(w))
        return -1 * np.sum(y * np.log(y_hat) + (1-y) * np.log(1 - y_hat)) / len(y)

    def _dJ(self, w, X, y):
        y_hat = self._sigmoid(X.dot(w))
        return X.T.dot(y_hat - y) / len(y)

    def _GD(self, X, y, init_w, lr, n_iters, epsilon):
        w = init_w
        cur_iter = 0

        while cur_iter < n_iters:
            last_J = self._J(w, X, y)
            gradient = self._dJ(w, X, y)
            w = w - lr * gradient
            J = self._J(w, X, y)
            if abs(J - last_J) < epsilon:
                break
            cur_iter += 1
        return w

    def fit(self, X_train, y_train, lr=0.01, n_iters=1e4, epsilon=1e-8):
        X = np.hstack([X_train, np.ones((len(X_train), 1))])
        init_w = np.zeros(X.shape[1])

        self._w = self._GD(X, y_train, init_w, lr, n_iters, epsilon)
        self.intercept_ = self._w[0]
        self.coef_ = self._w[1:]

        return self 

    def predict(self, X_predict):
        X = np.hstack([X_predict, np.ones((len(X_predict), 1))])
        return self._sigmoid(X.dot(self._w))


if __name__ == "__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X = X[y<2,:]
    y = y[y<2]
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_ratio=0.4)

    LRC = LogisticRegressionClassifier()
    LRC = LRC.fit(X_train, y_train, n_iters=100)
    predict = LRC.predict(X_test)
    print(predict)
    print(y_test)
    y_predict = predict > 0.5
    accuracy = np.sum(y_predict == y_test) / y_test.shape[0]
    print(accuracy)
