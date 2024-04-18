'''
ISYE 6740
Project
LASSO Implementation
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, Lasso
import random

# ---------------------------------
# -----Pre-process data-----
# ---------------------------------
data = pd.DataFrame(pd.read_csv('../data/dataset.csv'))
data = data.dropna()  # Removes all rows with 'nan'
feature_names = data.columns.tolist()
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
# -----LASSO Regression-----
# ---------------------------------
model = LassoCV(cv=5, random_state=0, max_iter=10000)
model.fit(Xtrain, ytrain.values.ravel())

lasso_best = Lasso(alpha=model.alpha_)
lasso_best.fit(Xtrain, ytrain.values.ravel())

y_pred = lasso_best.predict(Xtest)

mse = mean_squared_error(ytest, y_pred)
print(f'Mean Squared Error: {mse}')

print('R squared training set: ', round(lasso_best.score(Xtrain, ytrain.values.ravel())*100, 2))
print('R squared test set: ', round(lasso_best.score(Xtest, ytest.values.ravel())*100, 2))

print(f'The LASSO regression model predicted: {y_pred[4]}')
print(f'The actual value for the same given test case is: {ytest.iloc[4].values}')