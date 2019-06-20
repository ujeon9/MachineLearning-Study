from matplotlib import pyplot as plt
import numpy as np

a=np.random.randint(0,10,size=10000)
cnt=np.array([0,0,0,0,0,0,0,0,0,0])
num=0
for i in range(0,10):
    for j in range(0,10000):
        if num==a[j]:
            cnt[i]=cnt[i]+1
    num=num+1

num=0

#Count occurrence for each value in range
for i in range(0,10):
    print(num,":",cnt[i])
    num=num+1

number=['0','1','2','3','4','5','6','7','8','9']

#Draw pie char
plt.pie(cnt,labels=number,autopct='%1.2f%%')
plt.show()
