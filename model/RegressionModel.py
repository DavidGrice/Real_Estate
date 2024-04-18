import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import random

# ---------------------------------
# -----Pre-process data-----
# ---------------------------------
data = pd.DataFrame(pd.read_csv('dataset.csv'))
# print(data.dtypes)

X = data.iloc[:, 5:-1]
y = data.iloc[:, -1:]
# print(len(X.columns))
# print(y.columns)

scale = StandardScaler()
X = scale.fit_transform(X)

random.seed(79)
Xtrain, Xtest, ytrain, ytest, = train_test_split(X, y, test_size=0.3, random_state=79, shuffle=False)

# ---------------------------------
# -----RF Regression-----
# ---------------------------------
model = RandomForestRegressor(n_estimators=47, random_state=79, oob_score=True)
model.fit(Xtrain, ytrain.values.ravel())

y_pred = model.predict(Xtest)
# print(len(y_pred))

oob_score = model.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

mse = mean_squared_error(ytest, y_pred)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(ytest, y_pred)
print(f'R-squared: {r2}')

print(y_pred[4])
print(ytest.iloc[4].values)
