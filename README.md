# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SANJEEVI J
RegisterNumber:  212222110040
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("Placement_Data.csv")
dataset
dataset=dataset.drop('sl_no',axis=1)
dataset=dataset.drop('salary',axis=1)
dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
dataset.dtypes
dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes
dataset
X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values
Y
theta = np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))
def gradient_descent (theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta
theta =  gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X): 
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print("Y_PRED:",y_pred)
print("Y:",Y)
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)
xnew = np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew = predict(theta,xnew)
print(y_prednew)

```

## Output:

#### Dataset:
<img src = "https://github.com/Adhithyaram29D/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393540/f0e5af66-87dc-4dd7-be43-c817ad0d6473" width="500">

#### Datatypes:
<img src = "https://github.com/Adhithyaram29D/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393540/992ffd1f-0f1d-4431-8b28-6952814db938" height="200">

#### Labelled Data:
<img src = "https://github.com/Adhithyaram29D/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393540/ffbf35c5-b7b1-4ac6-8d48-5017d201f2eb" width="400">

#### Dependent variable Y:
<img src = "https://github.com/Adhithyaram29D/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393540/a34516e4-b57f-41c4-9d94-d3d43f3dce92" width="400">
<br>

#### Accuracy:
<img src = "https://github.com/Adhithyaram29D/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393540/9b8ef2d9-8a67-41d2-a35c-b3c0234f0895" width="200">

#### Y_predicted & Y:
<img src = "https://github.com/Adhithyaram29D/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393540/63bf8995-3b82-4807-b9fc-490663a85612" width="400">

#### Y_prednew:
<img src = "https://github.com/Adhithyaram29D/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/119393540/6c0adeaf-b6a3-4cde-84ac-683ba144cace" width="300">

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
