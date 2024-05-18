# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Prepare your data

    Clean and format your data
    Split your data into training and testing sets

2.Define your model

    Use a sigmoid function to map inputs to outputs
    Initialize weights and bias terms

3.Define your cost function

    Use binary cross-entropy loss function
    Penalize the model for incorrect predictions

4.Define your learning rate

    Determines how quickly weights are updated during gradient descent

5.Train your model

    Adjust weights and bias terms using gradient descent
    Iterate until convergence or for a fixed number of iterations

6.Evaluate your model

    Test performance on testing data
    Use metrics such as accuracy, precision, recall, and F1 score

7.Tune hyperparameters

    Experiment with different learning rates and regularization techniques

8.Deploy your model

    Use trained model to make predictions on new data in a real-world application.

## Program:
```py
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: ARUN KUMAR SUKDEV CHAVAN
RegisterNumber: 212222230013


import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
### Initial data set:

![image](https://github.com/Yogeshvar005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497367/cfd0dcc8-a05a-4667-bf00-df32c726d6fe)
### Data info:

![image](https://github.com/Yogeshvar005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497367/1f784783-3223-41e3-b45a-f55168d37536)
### Optimization of null values:

![image](https://github.com/Yogeshvar005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497367/8003d2b0-c5ac-4c21-a612-b418d155a84e)
### Assignment of x and y values:

![image](https://github.com/Yogeshvar005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497367/0f8d9936-9839-49fb-9135-1629166a4dcc)
![image](https://github.com/Yogeshvar005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497367/6679ef6f-9dcf-4519-a7ce-80e914ea8d4d)
### Converting string literals to numerical values using label encoder:

![image](https://github.com/Yogeshvar005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497367/4046c60c-d6be-48ad-a10f-310eb1a02e4f)
### Accuracy:
![image](https://github.com/Yogeshvar005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497367/15dea803-21d5-470c-9e49-8b3693d99391)
### Prediction:
![image](https://github.com/Yogeshvar005/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/113497367/d833d4d1-004b-42ae-b790-dcd450b6651e)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
