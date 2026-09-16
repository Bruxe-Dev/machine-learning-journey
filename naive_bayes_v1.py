import pandas as pd 
import seaborn as sb 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

df = pd.read_csv("data/spam.csv",encoding="latin-1")[['v1',"v2"]]

print(f"Dataframe: {df.head()}")

df.columns = ['label','message']

X = df['message']
Y = df['label']

X_train,Y_train,X_test,Y_test = train_test_split(
    X,
    Y,
    train_size=0.2,
    random_state=42,
    stratify=y
)