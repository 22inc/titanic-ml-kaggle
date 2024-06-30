# Definition Legend
#
# fp = File Path
# td = Train Data
# tp = Test Point
# rtp = Rate Test Point
#
# acc : 75.358%

import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

fp = "input/train.csv"
fp1 = "input/test.csv"

td = pd.read_csv(fp)
print(td.head())

if 'Survived' in td.columns:
    td['Age'].fillna(td['Age'].median(), inplace=True)
    td['Embarked'].fillna(td['Embarked'].mode()[0], inplace=True)
    td['Fare'].fillna(td['Fare'].median(), inplace=True)
    
    td['Sex'] = td['Sex'].map({'male': 0, 'female': 1})
    td = pd.get_dummies(td, columns=['Embarked'])
    
    ft = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    ft += [col for col in td.columns if col.startswith('Embarked_')]
    X = td[ft]
    y = td['Survived']
    
    xt, xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(xt, yt)
    
    yp = model.predict(xv)
    
    acc = accuracy_score(yv, yp)
    print(f"Validation acc: {acc:.2%}")
    
    td = pd.read_csv(fp1)
    
    td['Age'].fillna(td['Age'].median(), inplace=True)
    td['Fare'].fillna(td['Fare'].median(), inplace=True)
    td['Embarked'].fillna(td['Embarked'].mode()[0], inplace=True)
    td['Sex'] = td['Sex'].map({'male': 0, 'female': 1})
    td = pd.get_dummies(td, columns=['Embarked'])
    
    for col in ft:
        if col not in td.columns:
            td[col] = 0
    
    xt = td[ft]
    td['Survived'] = model.predict(xt)
    
    td[['PassengerId', 'Survived']].to_csv("output/submission.csv", index=False)
else:
    print("Survived column  not in the training data.")