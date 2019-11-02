import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn import linear_model

height1=[167,170,163,160,178,177,175,171,163,169,165,181,175,170,181,177,172,168,160,175,173,158,158,158,175,160]
height2=[163, 177, 179, 168, 174, 176, 162, 172, 155, 157, 179, 155, 178, 165, 179, 163, 168, 170, 161, 167, 165, 183, 172, 175, 160, 189]

#Compute mean , variance, and standard deviation for each group
print("Group 1's mean = ",np.mean(height1) , ", variance = ",np.var(height1)," , standard deviation = ",np.std(height1))
print("Group 2's mean = ",np.mean(height2) , ", variance = ",np.var(height2)," , standard deviation = ",np.std(height2))

#Create boxplot
plotData=[height1,height2]
plt.boxplot(plotData)
plt.show();
