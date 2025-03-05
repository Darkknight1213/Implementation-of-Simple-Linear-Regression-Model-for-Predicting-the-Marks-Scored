# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries
2. Set variables for assigning dataset values
3. Import Linear regression from Sklearnr
4. Assign the points for representing the graph
5. predict the regressio for makes by using the representation of the graph
6. Compare the graphs and hence we obtained the Linear Regression for the given datas

## Program:

### 1. Importing the Standard Libraries
```python
# Program to implement the simple linear regression model for predicting the marks scored.
# Developed by: Mohamed Riyaz Ahamed
# RegisterNumber: 212224240092


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

# read csv file
df=pd.read_csv('/content/student_scores.csv')

# Displaying the content in datafile
print(df)
```
![image](https://github.com/user-attachments/assets/0df66068-31b0-4cfa-834d-493d432d2d0e)

---

### 2. Setting Variables for Assigning Dataset Values
```python
# Segregating data to variables
x = df.iloc[:,:-1].values
y = df.iloc[:,1].values

print(x)
print(y)
```

![image](https://github.com/user-attachments/assets/07290241-f6bf-4296-8667-d787f84dffca)

---

### 3. Training the Linear Regression Model & Predicting Results
```python
#splitting train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#import linear regression model and fit the model with the data
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

#displaying predicted values
print(y_pred)
#displaying actual values
print(y_test)
```

![image](https://github.com/user-attachments/assets/4bb08e1b-1f9b-46a8-8cfb-5c7602118bfa)

---

### 4. Plotting the Training Data Regression Line
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

---


### 5. Plotting the Testing Data Regression Line
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

---

### 6. Evaluating the Model (MSE, MAE, RMSE)
```python
#find mae,mse,rmse
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
