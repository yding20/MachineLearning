#!/usr/bin/python3

import numpy as np    
import matplotlib.pyplot as plt
import sys   

epochs = int(sys.argv[2])
capacity = float(sys.argv[4])
eta =0.1
#print(epochs, capacity)

def ReadData(path):
    data = open(path, "r")
    lines = []
    for line in data:
        lines.append(line)
    
    FeatureArrayPlus = np.zeros((len(lines),123))
    Label = []
        
    for j in range(0,len(lines)):
        RatioSet = lines[j].split()
        Label.append(int(RatioSet[0]))
        for i in range(1, len(RatioSet)):  
            SingleSet = RatioSet[i].split(':')
            position = int(SingleSet[0])
            FeatureArrayPlus[j, position-1] = 1
    return FeatureArrayPlus, Label

def Train(FeatureArray, Label, eta, epochs, capacity):
    weights = np.zeros(len(FeatureArray[0,:]))
    weight1 = 0
    for i in range(epochs):
        for j in range(0, len(FeatureArray[:,0])):
            DotProduct = np.dot(weights, FeatureArray[j,:])+1*weight1
            if DotProduct*Label[j] <= 1  :
                weights = weights - 1/len(FeatureArray[:,1])*eta*weights + capacity*eta*Label[j]*FeatureArray[j,:]
                weight1 = weight1+ capacity*eta*Label[j]*1
            else:
                weights = weights - 1/len(FeatureArray[:,1])*eta*weights
    return weights, weight1   
    
def Test(FeatureArraytest, Labeltest, eta, weight1, weights):
    Count = 0 
    correct = 0
    for j in range(0, len(FeatureArraytest[:,0])):
        DotProduct = np.dot( weights, FeatureArraytest[j,:])+weight1*1
        sign = np.sign(DotProduct)
        Count=Count+1
        if sign * Labeltest[j] > 0 :
            correct=correct+1
    Accuracy = correct/Count
    return Accuracy

#FeatureArray, Label = ReadData('/u/cs246/data/adult/a7a.train')  ### Reading the traning data 
FeatureArray, Label = ReadData('a7a.train')  ### Reading the traning data

weights, weight1 = Train (FeatureArray, Label, eta, epochs, capacity)  ### Traning using perceptron 
        
#FeatureArraytest, Labeltest = ReadData('/u/cs246/data/adult/a7a.test')  ### Reading the test data
FeatureArrayTrain, LabelTrain = ReadData('a7a.train')  ### Reading the test data
AccuracyTrain = Test(FeatureArrayTrain, LabelTrain, eta, weight1, weights)
  
#FeatureArraytest, Labeltest = ReadData('/u/cs246/data/adult/a7a.test')  ### Reading the test data
FeatureArrayTest, LabelTest = ReadData('a7a.test')  ### Reading the test data
AccuracyTest = Test(FeatureArrayTest, LabelTest, eta, weight1, weights)

#FeatureArraytest, Labeltest = ReadData('/u/cs246/data/adult/a7a.test')  ### Reading the test data
FeatureArrayDev, LabelDev = ReadData('a7a.dev')  ### Reading the test data
AccuracyDev = Test(FeatureArrayDev, LabelDev, eta, weight1, weights)

print('EPOCHS:', epochs)
print('CAPACITY:', capacity)
print('TRAINING_ACCURACY:', AccuracyTrain)
print('TEST_ACCURACY :', AccuracyTest) 
print('DEV_ACCURACY', AccuracyDev)
print('FINAL_SVM: ',  weight1,  *weights )
