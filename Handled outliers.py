import numpy as np
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
from sklearn.preprocessing import StandardScaler

df=pd.read_csv('C:/work/DS-minilab-2-dataset.csv') #read csv file
data=df[['Gender','Height','Weight']]

h= df.iloc[:, 1] #read height from file and store to array
w= df.iloc[:, 2] #read weight from file and store to array

#Replace wrong values to NULL value
for i in range(0, len(h)):
    if h[i] > (h.mean() + (h.std()) * 2):
        h = h.replace(h[i], np.NaN)

    elif h[i] < (h.mean() - (h.std()) * 2):
        h = h.replace(h[i], np.NaN)
h=h.fillna(value=h.mean()) #Fill replaced NULL value to height's mean value

#Replace wrong values to NULL value
for i in range(0, len(w)):
    if w[i] > (w.mean() + (w.std()) * 2):
        w = w.replace(w[i], np.NaN)
    elif w[i] < (w.mean() - (w.std()) * 2):
        w =w.replace(w[i], np.NaN)
w=w.fillna(value=w.mean()) #Fill replaced NULL value to weight's mean value

#Change to numpy array for linear regression
x=np.array(h)
y=np.array(w)

reg=linear_model.LinearRegression()
reg.fit(x[:,np.newaxis],y) #Calculates a model on a training dataset
px=np.array([x.min()-1, x.max()+1])
py=reg.predict(px[:,np.newaxis]) #Prediction of output data for new input data

plt.scatter(x,y)
plt.title('Perform imputation of wrong and missing values using mean values')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.plot(px,py,color='r')
plt.show()
