import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import load_diabetes
from sklearn.base import BaseEstimator,RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV

grid_params = {
    "estimators": [5,10,15,20,25,30,35],
    "max_depth": [1,3,5,10,15, None]
}

class BaggingRegressor(BaseEstimator,RegressorMixin):
    def __init__(self,n_estimators,base_regressor,max_depth=None):
        self.base_regressor = base_regressor,
        self.n_estimators = n_estimators,
        self.max_depth = max_depth,
        self.regressors = []

    def fit(self,X,Y):
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X),len(X),replace=True)

            x_sampled,y_sampled = X[indices],Y[indices]

            clf = self.base_regressor(
                max_depth = self.max_depth
            )

            clf.fit(x_sampled,y_sampled)
            self.regressors.append(clf)

        return self.regressors

    def predict(self,X):
        predictions = np.array([clf.predict(X) for clf in self.regressors])

        major_votes = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),axis=0,err=predictions
        )

        return major_votes


if __name__ == "__main__":
    data = load_diabetes()
    labels = data.target_names

    X,y = data.data,digits.targs 

    x_train,x_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    grid_search = GridSearchCV(
        BaggingRegressor(
            n_estimators=5,
            base_regressor=DecisionTreeRegressor(),
        ),
        param_grid,
        cv=10,
        scoring="accuracy",
        verbose=1
    )

    grid_search.fit(x_train,y_train)

    print(f"Best params: {grid_search.best_params_}")
    print(f"Best CV Accuracy: {grid_search.best_score_}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(x_test)

    print(f"Test Accuracy: {accuracy_score(y_test,y_pred)}")

    for i, clf in enumerate(grid_search.best_estimator_.regressors):
        y_pred_i = clf.predict(x_test)
        acc_score_i = accuracy_score(y_test,y_pred_i)

        print(f"Accuracy of {i+1} regressor is: {acc_score_i}")