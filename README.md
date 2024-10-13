# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Get the data and use label encoder to change all the values to numeric. Drop the unwanted values,Check for NULL values, Duplicate values. Classify the training data and the test data. Calculate the accuracy score, confusion matrix and classification report.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: BHAVATHARANI S
RegisterNumber:  212223230032
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])

data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
*/

```

## Output:
![alt text](<Screenshot 2024-10-13 101331.png>)
![alt text](<Screenshot 2024-10-13 101342.png>)
![alt text](<Screenshot 2024-10-13 101352.png>)
![alt text](<Screenshot 2024-10-13 101402.png>)
![alt text](<Screenshot 2024-10-13 101407.png>)
![alt text](<Screenshot 2024-10-13 101414.png>)
![alt text](<Screenshot 2024-10-13 101421.png>)
![alt text](<Screenshot 2024-10-13 101421-1.png>)
![alt text](<Screenshot 2024-10-13 101434.png>)
![alt text](<Screenshot 2024-10-13 101442.png>)
![alt text](<Screenshot 2024-10-13 101448.png>)
![alt text](<Screenshot 2024-10-13 101502.png>)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
