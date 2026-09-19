import pandas as pd 
import seaborn as sb 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

# vectorizer2 = CountVectorizer()

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

df = pd.read_csv("data/spam.csv",encoding='latin-1')[['v1','v2']]

df.columns = ['label','message']

X = df['message']
Y = df['label'].map({'ham':0,"spam":1})

X_train,X_test,Y_train,Y_test = train_test_split(
    X,
    Y,
    train_size=0.8,
    random_state=42,
    stratify=Y
)

pipeline = Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb', MultinomialNB())
])

#X_train_vectors = pipeline.fit(X_train,Y_train)
#X_test_vectors = pipeline.transform(X_test)


pipeline.fit(X_train,Y_train)
y_pred = pipeline.predict(X_test)

scores = cross_val_score(
    pipeline,
    X_train,
    Y_train,
    cv=skf
)

for score in scores:
    print(f"Score: {score*100:.2f}% \n")

mean_score = np.sum(scores)/ 5

accuracy = accuracy_score(Y_test,y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

cm = confusion_matrix(Y_test,y_pred)
sb.heatmap(cm,annot=True,fmt='d',cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])

plt.title("Connfusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

print(classification_report(Y_test, y_pred))


new_messages = [
    "Congratulations! You got 95 in your math exam.",
    "Hey, are we still meeting today?"
]

new_vec = pipeline.transform(new_messages)
predictions = pipeline.predict(new_vec)

for msg, pred in zip(new_messages, predictions):
    label = "Spam" if pred == 1 else "Ham"
    print(f"Message: '{msg}' => Prediction: {label}")