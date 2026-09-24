import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline

param_grid = {
    "n_estimators": [5,10,15,20,25,30],
    "max_depth": [1,5,10,15,None]
}

class BaggingClassifier(BaseEstimator,ClassifierMixin):
    def __init__(self,base_classifier,n_estimators,max_depth=None):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.classifiers = []

    def fit(self,X,Y):
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X),len(X),replace=True)

            x_sampled,y_sampled = X[indices],Y[indices]
            clf = self.base_classifier.__class__(
                max_depth = self.max_depth
            )
            clf.fit(x_sampled,y_sampled)

            self.classifiers.append(clf)

        return self.classifiers

    def predict(self,X):
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        majority_votes = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(),axis=0,arr=predictions
        )

        return majority_votes

if __name__ == "__main__":
    digits = load_digits()
    labels = digits.target_names

    X,y = digits.data,digits.target

    x_train,x_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    #base_clf = DecisionTreeClassifier()

    grid_src = GridSearchCV(
        BaggingClassifier(
            base_classifier=DecisionTreeClassifier(),
            n_estimators=10
        ),
        param_grid,
        cv=5,
        scoring="accuracy"
    )
    #model = BaggingClassifier(base_classifier=base_clf,n_estimators=20)
    grid_src.fit(x_train,y_train)

    print("Best Parameters:\n", grid_src.best_params_)
    print("Best CV Accuracy:\n", grid_src.best_score_)

    best_model = grid_src.best_estimator_
    y_pred = best_model.predict(x_test)
 
    print(f"Overall Accuracy: {accuracy_score(y_test,y_pred)*100:.2f}%\n\n")

    for i, clf in enumerate(grid_src.best_estimator_.classifiers):
        y_pred_i = clf.predict(x_test)
        acc_score_i = accuracy_score(y_test,y_pred_i)

        print(f"Accuracy of classifier {i+1} = {acc_score_i:.4f}")

    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(8,6))

    sb.heatmap(
        cm,
        annot=True,
        cmap="Blues",
        xticklabels= labels,
        yticklabels= labels
    )

    plt.ylabel("Actual Labels")
    plt.xlabel("Predicted Labels")
    plt.title("Confusion Matrix")
    plt.show()      