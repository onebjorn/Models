import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score



TREE_PARAMS_DICT = {'max_depth': 7}


class GradientBoostingClassifier(BaseEstimator):
    
    def __init__(self, tree_params, n_iterations, learning_rate):
        self.tree_params = tree_params
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y):

        self.base_algo = DecisionTreeRegressor(**self.tree_params).fit(X, y)
        self.estimators = []

        curr_pred = self.base_algo.predict(X)

        for _ in range(self.n_iterations):
            # y это 0 или 1
            # a - сырое предсказание
            # f(a) = 1 / (1 + exp(-a)) - преобразование в вероятность
            # f'(a) = - exp(a) / (1 + exp(-a))^2 = - f(a) (1 - f(a))
            # log loss это (y log f(a) + (1 - y) log(1 - f(a)))

            # d/da (y log f(a) + (1 - y) log(1 - f(a))) = f'(a) (y/f(a) - (1 - y) / (1 - f(a)))
            fa = 1. / (1 + np.exp(-curr_pred))
            grad = - fa * (1. - fa) * (y / fa - (1. - y) / (1. - fa))

            algo = DecisionTreeRegressor(**self.tree_params).fit(X, -grad)

            self.estimators.append(algo)

            curr_pred += self.learning_rate * algo.predict(X)

        return self


    def predict_proba(self, X):

        res = self.base_algo.predict(X)

        for estimator in self.estimators:
            res += self.learning_rate * estimator.predict(X)
        
        return res

    
    def predict(self, X):

        res = self.base_algo.predict(X)

        for estimator in self.estimators:
            res += self.learning_rate * estimator.predict(X)

        return list(map(lambda x: 1 if x > 0.5 else 0, res))


if __name__ == '__main__':

    X_data, y_data = make_classification(
        n_samples=10000, n_features=20, 
        n_classes=2, n_informative=20, 
        n_redundant=0,random_state=42
    )

    X_train, X_test, Y_train, Y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    model = GradientBoostingClassifier(tree_params=TREE_PARAMS_DICT, learning_rate=0.07, n_iterations=100)

    model.fit(X_train, Y_train)

    print(accuracy_score(Y_test, model.predict(X_test)))
    print(roc_auc_score(Y_test, model.predict_proba(X_test)))
