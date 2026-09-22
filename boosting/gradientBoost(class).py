import matplotlib.pyplot as plt 
import seaborn as sb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_digits

X,Y = load_digits(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(
    X,
    Y,
    random_state=42,
    test_size=0.25
)

gbc = GradientBoostingClassifier(n_estimators=300,learning_rate=0.05,max_features=5,random_state=100)

gbc.fit(X_train,y_train)

pred_y = gbc.predict(X_test)

acc = accuracy_score(y_test,pred_y)
cm = confusion_matrix(y_test,pred_y)

print(f"The Accuracy score is: {acc}")

plt.figure(figsize=(8,6))

sb.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.ylabel("Actual Labels")
plt.xlabel("Predicted Labels")
plt.title("Confusion Matrix")
plt.show()