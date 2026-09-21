import pandas as pd 
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import warnings

warnings.filterwarnings('ignore')
categorical_trans = OneHotEncoder(handle_unknown="ignore")

imputer = SimpleImputer(strategy="median")

titanic_data = pd.read_csv("../data/titanic.csv")
titanic_data = titanic_data.dropna(subset=["Survived"])

x = titanic_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = titanic_data['Survived']

param_grid ={
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [5, 10, None]
}

numerical_features = [
    'Pclass',
    'Age',
    'SibSp',
    'Parch',
    'Fare'
]
categorical_features = ["Sex"]

preprocessing = ColumnTransformer(
    transformers = [
        ("num", imputer, numerical_features),
        ("cat",categorical_trans,categorical_features)
    ]
)

X_train,X_test,Y_train,Y_test = train_test_split(
    x,
    y,
    random_state=42,
    test_size=0.2
)

# X_train['Sex'] = hotEncoder.fit_transform(X_train[['Sex']])
# X_train['Age'] = X_train['Age'].fillna(X_train['Age'].median())4

pipeline = Pipeline([
    ("preprocessor",preprocessing),
    ("classifier", RandomForestClassifier()) 
])

#rf_classifier = RandomForestClassifier(n_estimators=50,random_state=42)
pipeline.fit(X_train,Y_train)

y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(Y_test,y_pred)
classification_rep = classification_report(Y_test,y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

sample = X_test.iloc[0:1]
prediction = pipeline.predict(sample)

sample_dict = sample.iloc[0].to_dict()
print(f"\nSample Passenger: {sample_dict}")
print(f"Predicted Survival: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")