import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import linear_model
import math

height = [[167],[170],[163],[160],[178],[177],[175],[171],[163],[169],[165],[181],[175],[170],[181],[177],[172],[168],[160],[175],[173],[158],[158],[158],[175],[160],[163], [177], [179], [168], [174], [176], [162], [172], [155], [157], [179], [155], [178], [165], [179], [163], [168], [170], [161], [167], [165], [183], [172], [175], [160], [189]]
weight= [55,72,54,48,64,70,79,76,60,84,57,69,66,75,72,64,71,45,55,58,73,49,47,50,79,56,48, 66, 70, 63, 60, 70, 49, 70, 45, 66, 70, 56, 70, 44, 55, 49, 58, 72, 45, 57, 48, 90, 72, 73, 50, 75]

reg=linear_model.LinearRegression() #Create linear regression model
reg.fit(height,weight) #Learn data

B=reg.coef_[0] #Get slope
A=reg.intercept_ #Get y-axis intercept

print("X = ",height)
print("Y = ",weight)
print("Linear regression (y = Bx + A): ")
print("B = ",B,"A = ",A)

plt.scatter(weight,height)

plt.xlabel("weight")
plt.ylabel("height")

plt.show();
