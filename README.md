# Simple-Linear-Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data_lr.csv')
X = data['Height'].values.reshape(-1, 1)
y = data['Weight'].values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Plotting the training data and the model
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression line')
plt.title('Height vs Weight (Training set)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.show()

# Plotting the test data and the model
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X_train, model.predict(X_train), color='red', label='Regression line')
plt.title('Height vs Weight (Test set)')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend()
plt.show()
