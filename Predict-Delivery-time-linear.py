import pandas as pd
import numpy as np
import io
import warnings
warnings.filterwarnings(action='ignore')
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

df=pd.read_csv('C://work\linear_regression_data.csv', encoding='utf-8')

iris_dataset=df

#split dataset into train and test data
x_train, x_test, y_train, y_test=train_test_split(iris_dataset['Distance'],iris_dataset['Delivery Time'],test_size=0.2,random_state=33)


x_train=x_train.values.reshape(-1,1)
y_train=y_train.values.reshape(-1,1)

x_test=x_test.values.reshape(-1,1)
y_test=y_test.values.reshape(-1,1)

# Create LinearRegression
lm=LinearRegression()
# Fit Data
lm.fit(x_train,y_train)

# Predictions on the test data
px=np.array([x_train.min()-1, x_train.max()+1])
py=lm.predict(px[:,np.newaxis])
predictions=lm.predict(x_test)
plt.scatter(x_test,predictions)

# Draw linear regression 
plt.plot(px,py,color='r')
plt.title('Predict Delivery time')
plt.xlabel('Distance')
plt.ylabel('Delivery Time')
plt.show()

#Evaluate model
print("Test Acuuracy is ",format(lm.score(x_test,y_test)))
                            
