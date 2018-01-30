#!/usr/bin/python3

import numpy as np        
import sys

iterations = int(sys.argv[3])

def ReadData(path):
    data = open(path, "r")
    lines = []
    for line in data:
        lines.append(line)
    
    FeatureArrayPlus = np.zeros((len(lines),124))
    Label = []
        
    for j in range(0,len(lines)):
        RatioSet = lines[j].split()
        Label.append(int(RatioSet[0]))
        for i in range(1, len(RatioSet)):  
            SingleSet = RatioSet[i].split(':')
            position = int(SingleSet[0])
            FeatureArrayPlus[j, position] = 1
            FeatureArrayPlus[j, -1] = 1
    return FeatureArrayPlus, Label

def Dev(FeatureArray, Label, eta, iterations):
    weights = np.zeros(len(FeatureArray[0,:]))
    for i in range(iterations):
        for j in range(0, len(FeatureArray[:,0])):
            DotProduct = np.dot(weights, FeatureArray[j,:])
            sign = np.sign(DotProduct)
            if sign!= Label[j]:
                weights = weights + eta*Label[j]*FeatureArray[j,:]
    return weights  

def Train(FeatureArray, Label, eta, iterations, Devweights):
    weights = Devweights
    for i in range(iterations):
        for j in range(0, len(FeatureArray[:,0])):
            DotProduct = np.dot(weights, FeatureArray[j,:])
            sign = np.sign(DotProduct)
            if sign!= Label[j]:
                weights = weights + eta*Label[j]*FeatureArray[j,:]
    return weights   
    
def Test(FeatureArraytest, Labeltest, eta):
    Count = 0 
    right = 0
    for j in range(0, len(FeatureArraytest[:,0])):
        DotProduct = np.dot( weights, FeatureArraytest[j,:])
        sign = np.sign(DotProduct)
        Count=Count+1
        if sign == Labeltest[j] :
            right=right+1
    Accuracy = right/Count
    return Accuracy


#FeatureArray, Label = ReadData('/u/cs246/data/adult/a7a.train')  ### Reading the traning data 
FeatureArray, Label = ReadData('a7a.train')  ### Reading the traning data

eta =1
if sys.argv[1] == '--nodev':
    Devweights = np.zeros(len(FeatureArray[0,:]))
else:
    Devweights = Dev (FeatureArray, Label, eta, iterations)  
weights = Train (FeatureArray, Label, eta, iterations, Devweights)  ### Traning using perceptron 
        
#FeatureArraytest, Labeltest = ReadData('/u/cs246/data/adult/a7a.test')  ### Reading the test data
FeatureArraytest, Labeltest = ReadData('a7a.test')  ### Reading the test data
Accuracy = Test(FeatureArraytest, Labeltest, eta)
  
print('Test accuracy :' , Accuracy) 
print('Feature weights (bias last) : ',  *weights) 
