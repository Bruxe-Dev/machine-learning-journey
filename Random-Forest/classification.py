import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import warnings

warnings.filterwarnings('ignore')
le = LabelEncoder()

titanic_data = pd.read_csv("../data/titanic.csv")
titanic_data = titanic_data.dropna(subset=["Survived"])

x = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']

x['Sex'] = le.fit_transform(x['Sex'])

X_train,X_test,Y_train,Y_test = train_test_split(
    x,
    y,
    random_state=42,
    test_size=0.2
)

X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())

rf_classifier = RandomForestClassifier(n_estimators=50,random_state=42)
rf_classifier.fit(X_train,Y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(Y_test,y_pred)
classification_rep = classification_report(Y_test,y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

sample = X_test.iloc[0:1]
prediction = rf_classifier.predict(sample)

sample_dict = sample.iloc[0].to_dict()
print(f"\nSample Passenger: {sample_dict}")
print(f"Predicted Survival: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")