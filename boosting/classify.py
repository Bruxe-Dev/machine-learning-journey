import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_recall_curve,recall_score,roc_auc_score
from sklearn.datasets import load_iris,make_classification
import warnings 

warnings.filterwarnings('ignore')

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = []
        self.alphas = []

    def fit(self,X,Y):
        n_samples,n_features = X.shape()
        w = np.ones(n_features)/n_features

        for _ in range(self.n_estimators):
            model = DecisionTreeClassifier(max_depth=2)
            model.fit(X,Y,sample_weight=w)
            predictions = model.predict(X)

            err = np.sum(w * (predictions != Y)) / np.sum(w)
            alpha = 0.5 * np.log(1 - err / err * 1e-10)

            self.models.append(model)
            self.alphas.append(alpha)

            w *= np.exp(-alpha * y * predictions)
            w /= np.sum(w)

    def predict(self,X):
        strong_preds = np.zeros(X.shape[0])

        for model, alpha in zip(self.models,self.alphas):
            prediction = model.predict(X)
            strong_preds += alpha * prediction

        return np.sign(strong_preds).astype(int)


if __name__ == "__main__":
    X,Y = make_classification(n_samples=1000,n_features=20,n_classes=2,random_state=42)
    X_train,X_test,y_train,y_test = train_test_split(
        X,
        Y,
        test_size=0.3,
        random_state=42
    )

    adaboost = AdaBoost(n_estimators=50)
    adaboost.fit(X_train,y_train)

    predictions = adaboost.predict(X_test)

    accuracy = accuracy_score(y_test,predictions)
    precision = recall_score(y_test,predictions)
    f1_score = f1_score(y_test,predictions)

    try:
        roc_auc = roc_auc_score(y_test,predictions)

    except ValueError:
        print(f"Underfined (Expects probability score!)")

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"f1_Score: {f1_score}")
    print(f"ROC_AUC: {roc_auc}")