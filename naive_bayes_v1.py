import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sb 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

df = pd.read_csv("data/spam.csv",encoding='latin-1')[['v1','v2']]

print(df.head())

df.columns = ['label','message']

print(df.head())
X = df['message']
Y = df['label']

print(Y.value_counts())

X_vector = vectorizer.fit_transform(X)
print(vectorizer.get_feature_names_out())
print(X_vector.shape)

X_train,Y_train,X_test,Y_test  = train_test_split(
    X_vector,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print(f"Training: ",X_train.shape[0])
print(f"Testing: ",X_test.shape[0])
print("\n Training Distribution: ")
print(Y_train.value_counts())
print("\n Test Distribution: ")
print(Y_test.value_counts()) 