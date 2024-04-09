'''
ISYE 6740
Project
Decision Tree Classifier
'''

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# load in data
data = pd.read_csv('../02_Data/NC_2022-2024_redfin_data.csv', engine='python')

print(data)

grouped_data = data.groupby('Region')
regions = data['Region'].unique()


Y = data['Average Sale To List']
X = data['Month of Period End']

dt_classifier = DecisionTreeClassifier()
