import numpy as np
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd

df=pd.read_excel('C:/work/DS-lab-3-dataset.xlsx') #read excel file
data=df[['Gender','Height','Weight','Index']]

#read gender, height, weight from file and store to array
g=data.iloc[:,0]
h=data.iloc[:,1]
w=data.iloc[:,2]
BMI=data.iloc[:,3]

a=np.array(h)
b=np.array(w)
c=np.array(g)
a=a.reshape((100,1))
b=b.reshape((100,1))
c=c.reshape((100,1))

BMI=np.array(BMI)
BMI=BMI.reshape((100,1))

#Arrays for divide into two groups(Female, Male)
af=np.array(BMI)
bf=np.array(BMI)
am=np.array(BMI)
bm=np.array(BMI)


reg=linear_model.LinearRegression() 
reg.fit(a,b) #Calculates a model on a training dataset
y=reg.intercept_ #y intercept
k=reg.coef_[0] #slope

#Store for e=w-w'
e=np.arange(0,100)

#Calculate e=w-w using y intercept(y), slope(k), height(a)
for i in range(0, 100):
    e=k*a+y 

z=np.arange(0,100)

#Calculate ze=[e-μ(e)]/σ(e),
for i in range(0, 100):
    z=(e-e.mean())/e.std()
    
m=z.mean()

plt.hist(z,bins=5, histtype='bar',rwidth=0.9)
plt.title('Histogram of after handling outlier people')
plt.xlabel('z')
plt.ylabel('number of people')
plt.show()

#If z<-(z.mean()+0.25), set BMI=0; if z<z.mean+0.25, set BMI=5
t=m+0.25

for i in range(0, 100):
    if -(t)>z[i]:
        BMI[i]=0
    elif t<z[i]:
        BMI[i]=5

print(BMI)

cnt=0 #For counting Female and Male number

for i in range(0, 100):
    if c[i]=='Female':
        cnt=cnt+1
for i in range(0, 100):
    if c[i]=='Female':
          af[i]=a[i]
          bf[i]=b[i]
    else:
          am[i]=a[i]
          bm[i]=b[i]
          
#Change to DataFrame to using dropna()
af=pd.DataFrame(af)
af=af.dropna()
bf=pd.DataFrame(bf)
bf=bf.dropna()

#Change to numpy array to using linear regression
af=np.array(af)
bf=np.array(bf)

reg=linear_model.LinearRegression()
reg.fit(af,bf) #Calculates a model on a training dataset
yf=reg.intercept_ #y intercept
kf=reg.coef_[0] #slope

#Store for e=w-w'
ef=np.arange(0,cnt)

#Calculate ef=wf-wf using yf intercept(yf), slope(kf), height(af)
for i in range(0,cnt):
    ef=kf*af+yf
    
zf=np.arange(0,cnt)

#Calculate ze=[e-μ(e)]/σ(e),
for i in range(0,cnt):
    zf=(ef-ef.mean())/ef.std()
    
plt.hist(zf,bins=5,histtype='bar',rwidth=0.9)
plt.title('Histogram of after handling outlier Female')
plt.xlabel('z')
plt.ylabel('number of people')
plt.show()

#Change to DataFrame to using dropna()
am=pd.DataFrame(am)
am=am.dropna()
bm=pd.DataFrame(bm)
bm=bm.dropna()

#Change to numpy array to using linear regression
am=np.array(am)
bm=np.array(bm)

reg=linear_model.LinearRegression()
reg.fit(am,bm) #Calculates a model on a training dataset
ym=reg.intercept_ #y intercept
km=reg.coef_[0] #slope

cnt2=100-cnt #For store Male number

#Store for e=w-w'
em=np.arange(0,cnt2)

#Calculate ef=wf-wf using yf intercept(yf), slope(kf), height(af)
for i in range(0,cnt2):
    em=km*am+ym
    
zm=np.arange(0,cnt2)

#Calculate ze=[e-μ(e)]/σ(e),
for i in range(0,cnt2):
    zm=(em-em.mean())/em.std()

plt.hist(zm,bins=5, histtype='bar',rwidth=0.9)
plt.title('Histogram of after handling outlier Male')
plt.xlabel('z')
plt.ylabel('number of people')
plt.show()

