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