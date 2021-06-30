# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

data=pd.read_csv('diabetes.csv')

y=data[['Outcome']]
x=data[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]

x.shape

x.head(5)

import seaborn as sns
import matplotlib.pyplot as plt
corrmat=data.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(15,15))
g=sns.heatmap(data[top_corr_features].corr(),annot = True,cmap='RdYlGn')

data.corr()

def fillskinthickness(row):
    if not((row['SkinThickness'])==0):
        return row['SkinThickness']
    if row['BMI']>=0 or row['BMI']<=100:        
        if row['BMI']>=30 or row['BMI']<40:
            return 35
        elif row['BMI']>=20 or row['BMI']<30:
            return 25
        elif row['BMI']>=40 or row['BMI']<50:
            return 45
        elif row['BMI']>=50 or row['BMI']<60:
            return 55

x['SkinThickness']=x.apply(fillskinthickness, axis=1)

x['SkinThickness']

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x=imputer.fit_transform(x)
y = imputer.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y,test_size=0.1,random_state=0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponent = pca.fit_transform(x_train)
principalComponent = pca.fit_transform(x_test)

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=150,max_depth=100)

classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10)

y_pred = classifier.predict(x_test)

accuracies.mean()

accuracies

acc_decision_tree = round(classifier.score(x_train, y_train) * 100, 2)
acc_decision_tree

from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)

from xgboost import XGBClassifier
model = XGBClassifier()

classifier.fit(x_train, y_train)

y_pred1=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test,y_pred1)
print(confusion_matrix)

