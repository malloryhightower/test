import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

data=pd.read_csv('dataset.csv')

data=data.drop(['Unnamed: 5', 'Unnamed: 6','Index'], axis=1)

X = data.iloc[:, :3]
y = data.iloc[:, 3]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Fitting model with trainig data
regressor.fit(X, y)

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1,10,9]]))

