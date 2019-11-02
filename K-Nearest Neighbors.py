from numpy import *
import operator

def createDataSet():
    #group=[height, weight] labels=[size]
    group=array([[158,58],[158,59],[158,63],[160,59],[160,60],[163,60],[163,61],[160,64],[163,64],[165,61],
                 [165,62],[165,65],[168,62],[168,63],[168,66],[170,63],[170,64],[170,68]])
    labels=['M','M','M','M','M','M','M','L','L','L','L','L','L','L','L','L','L','L']
    
    return group, labels

#For calculates the distance between inX and training data, and lists which is closest.
def kNN(inX,trainingData,labels,k):
    dataSetSize=trainingData.shape[0] #Number of training data to use
    
    #The difference between the values to classify and the training data 
    diff=tile(inX,(dataSetSize,1))-trainingData
    
    sqDiff=diff**2                                
    sqDistance=sqDiff.sum(axis=1)
    distance=sqDistance**0.5
    sortedDist=distance.argsort() #Sort indexes in order that calculated distance values increase
    classCount={}   #Declared to gather and explore classes of the three closest data
    
    for i in range(k):
        candidateLabel=labels[sortedDist[i]]
        classCount[candidateLabel]=classCount.get(candidateLabel,0)+1

    sortedClassCount=sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#Class prediction
group,labels=createDataSet()
print(kNN(array([[1,1]]),group,labels,3))

	





















