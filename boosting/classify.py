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