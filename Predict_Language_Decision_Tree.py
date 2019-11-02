import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings(action='ignore')
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score

df=pd.read_csv('C://work\decision_tree_data.csv',encoding='utf-8')

tmp=[]
tmp1=[]
tmp2=[]
tmp3=[]
tmp4=[]

#Preprocessing data
for each in df['level']:
    if each=='senior':
        tmp.append(2)
    elif each == 'mid':
        tmp.append(1)
    elif each == 'junior':
        tmp.append(0)
        
for each in df['lang']:
    if each=='java':
        tmp1.append(2)
    elif each == 'python':
        tmp1.append(1)
    elif each == 'R':
        tmp1.append(0)
for each in df['tweets']:
    if each=='no':
        tmp2.append(0)
    elif each == 'yes':
        tmp2.append(1)
for each in df['phd']:
    if each=='no':
        tmp3.append(0)
    elif each == 'yes':
        tmp3.append(1)

df['interview']=df['interview'].astype('int')
df['level']=tmp
df['lang']=tmp1
df['tweets']=tmp2
df['phd']=tmp3

#split dataset into train and test data
train_pre=df[['level','lang','tweets','phd']]
X_train, X_test, y_train, y_test = train_test_split(train_pre, df[['interview']], test_size=0.1, random_state=13)

# Create DecisionTree classifier
tree_clf = DecisionTreeClassifier(criterion='entropy',max_depth=3, random_state=13)
# Fit the classifier to the data
tree_clf.fit(X_train, y_train)

#show model predictions on the test data
y_pred=tree_clf.predict(X_test)

#Draw decision tree in png
export_graphviz(
        tree_clf,
        out_file="result.dot",
        feature_names=['level','lang','tweets','phd'],
        class_names=['False','True'],
        rounded=True,
        filled=True
    )

import graphviz
with open("result.dot") as f:
    dot_graph = f.read()
dot = graphviz.Source(dot_graph)
dot.format = 'png'
dot.render(filename='tree', directory='images/decision_trees', cleanup=True)
dot

#Evaluate model
print("Test Acuuracy is ", accuracy_score(y_test,y_pred)*100)

result=tree_clf.predict(X_test)
result=result.astype('bool')

print('result   : ',result)



