import pandas as pd 
import seaborn as sb 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.naive_bayes import MultinomialNB

vectorizer = CountVectorizer()

df = pd.read_csv("data/spam.csv",encoding="latin-1")[['v1',"v2"]]

print(f"Dataframe: {df.head()}")

df.columns = ['label','message']

X = df['message']
Y = df['label'].map({'ham':0,'spam':1})

X_train,Y_train,X_test,Y_test = train_test_split(
    X,
    Y,
    train_size=0.2,
    random_state=42,
    stratify=Y
)

x_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

nb_model = MultinomialNB()
nb_model.fit(x_train_vectors,Y_train)

y_pred = nb_model.predict(X_test_vectors)

accuracy = accuracy_score(Y_test,y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

cm = confusion_matrix(Y_test,y_pred)
sb.heatmap(cm,annot=True,fmt='d',cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])

plt.title("Connfusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

print(classification_report(y_test, y_pred))