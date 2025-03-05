# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:

### 1
```python
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Mohamed Riyaz Ahamed
RegisterNumber: 212224240092


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

df=pd.read_csv('/content/student_scores.csv')
print(df)
print(df.head())
print(df.tail())
```
![image](https://github.com/user-attachments/assets/d1217ac2-c9fa-400f-ac12-b0eecd855ec0)

### 2
```python
x = df.iloc[:,:-1].values
y = df.iloc[:,1].values

print(x)
print(y)
```

![image](https://github.com/user-attachments/assets/07290241-f6bf-4296-8667-d787f84dffca)

### 3
```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

print(y_pred)
print(y_test)
```

![image](https://github.com/user-attachments/assets/4bb08e1b-1f9b-46a8-8cfb-5c7602118bfa)

### 4
```python
#Graph plot for training data

plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')

plt.title("Hours vs Scores(Training set)")

plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

![image](https://github.com/user-attachments/assets/42df5909-d90a-4029-b66a-4fa5831cc1ef)

### 5
```python
#Graph plot for test data

plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')

plt.title("Hours vs Scores(Testing set)")

plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

![image](https://github.com/user-attachments/assets/3f8a56a9-9aaa-4e87-9e17-0fd7a71ee128)

### 6
```python
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

![image](https://github.com/user-attachments/assets/57d39ad2-2823-4a18-8811-2cf1396bf3e6)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
