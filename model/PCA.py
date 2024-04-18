'''
ISYE 6740
Project
PCA Implementation
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import decomposition
from sklearn.linear_model import LinearRegression
import random

def PCA(x, num_components):
    pca = decomposition.PCA(n_components=num_components)
    pca.fit(x)
    X = pca.transform(x)
    return X

# ---------------------------------
# -----Pre-process data-----
# ---------------------------------
data = pd.DataFrame(pd.read_csv('../data/dataset.csv'))
data = data.dropna()  # Removes all rows with 'nan'
# print(data.dtypes)

X = data.iloc[:, 5:-1]
y = data.iloc[:, -1:]
# print(len(X.columns))
# print(y.columns)

scale = StandardScaler()
X = scale.fit_transform(X)
num_comp = 2
X = PCA(X, num_components=num_comp)

random.seed(79)
Xtrain, Xtest, ytrain, ytest, = train_test_split(X, y, test_size=0.3, random_state=79, shuffle=False)

# ---------------------------------
# -----RF Regression-----
# ---------------------------------
model = LinearRegression()
model.fit(Xtrain, ytrain.values.ravel())

y_pred = model.predict(Xtest)
# print(len(y_pred))


mse = mean_squared_error(ytest, y_pred)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(ytest, y_pred)
print(f'R-squared: {r2}')

print(f'The {num_comp}-component PCA linear regression model predicted: {y_pred[4]}')
print(f'The actual value for the same given test case is: {ytest.iloc[4].values}')
