import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


class GradientBoostingHand:
    def __init__(self, gamma, n_estimators = 100, learning_rate = 0.01, random_state = 10):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.gamma = gamma

    def loss_gradient(self, y_true, y_pred):
        return y_true - y_pred

    def base_algoritm(self, y):
        return np.mean(y, axis = 0)

    def gamma_regul(self, tree):
        # штраф за количество лисьтев
        return self.gamma * tree.tree_.n_leaves

    def fit(self, X, y):
        self.trees = []
        self.y_base = self.base_algoritm(y)
        self.y_pred = self.y_base
        # Ансамбль состоит из n_estimators деревьев, каждый следующий алгоритм уменьшает ошибку предыдущего
        for i in range(self.n_estimators):
            gradient = self.loss_gradient(y, self.y_base)
            clf = DecisionTreeClassifier(random_state = self.random_state)
            # каждый новый алгоритм обучается на остаточных ошибках допущенным предыдущим прогнозатором
            clf.fit(X, gradient)
            self.trees.append(clf)
            # накапливаем остатки
            self.y_pred += self.learning_rate * clf.predict(X) + self.gamma_regul(clf)

    def predict(self, X):
        # к изначальному среднему значению добавляем остатки
        y_pred = self.y_base + self.learning_rate * np.sum(
            [tree.predict(X) for tree in self.trees], axis = 0) + np.sum(
            [self.gamma_regul(tree) for tree in self.trees], axis = 0)
        return y_pred






def overfitting(model, X_train, X_test, y_train, y_test):
    check_train = mean_squared_error(y_train, model.predict(X_train))
    check_test = mean_squared_error(y_test, model.predict(X_test))

    print(f'MSE train = {check_train}')
    print(f'MSE test = {check_test}')

    value = 1 - (check_train / check_test)
    return value


















































