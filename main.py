# This program was written in collaboration with students Pranav Damal, Andrew Cheng, and Aathithya Selvam
# Some of the code was from Jeff Spagnola, https://medium.com/swlh/decoding-customer-churn-6d1e941d0c7


import pandas as pd
import sklearn
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import preprocessing
from sklearn import utils


table = pd.read_csv('FINALIZED_dataset_rev1.csv')

lab_enc = preprocessing.LabelEncoder() 
unemployRate = table.pop('unemployRate')
y = lab_enc.fit_transform(unemployRate)

x = table.iloc[:,    : ]
x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, test_size=0.5)

model = RandomForestClassifier()
model.fit(x_train, y_train)

predictions = model.predict(x_test)
print(accuracy_score(y_test, predictions))

for i in range(1):
    x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, test_size=0.5)
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print("Accuracy score across 1 trial is: ")
    print(accuracy_score(y_test, predictions))

sklearn.metrics.plot_confusion_matrix(model, x_test, y_test)
feature_importance = pd.DataFrame({'Feature' : x_train.columns, 'Importance' : model.feature_importances_})
feature_importance.sort_values('Importance', ascending=False, inplace=True)

print(feature_importance)
pyplot.show()
