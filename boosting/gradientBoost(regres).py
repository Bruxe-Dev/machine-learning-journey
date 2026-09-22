import matplotlib.pyplot as plt 
import seaborn as sb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor 
import warnings 

warnings.filterwarnings("ignore")

X,Y = load_diabetes(return_X_y=True)

X_train,X_test,y_train,y_test = train_test_split(
    X,
    Y,
    random_state=42,
    test_size=32
)

gbr = GradientBoostingRegressor(
    n_estimators=300,
    loss="absolute_error",
    learning_rate=0.1,
    random_state=32,
    max_depth=1,
    max_features=5
    )

gbr.fit(X_train,y_train)
pred_y = gbr.predict(test_X)

test_rmse = mean_squared_error(test_y, pred_y) ** (1 / 2)

print('Root mean Square error: {:.2f}'.format(test_rmse))