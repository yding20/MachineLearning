#!/usr/bin/python3

import numpy as np    
import matplotlib.pyplot as plt
import argparse    

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', action='store', dest='epochs', type=int, default=1)
#parser.add_argument('--nodev', action='store_true', dest='noDevCriterion', default=False)
parser.add_argument('--capacity', action='store', dest='capacity', type=float, default=1)
argu = parser.parse_args()

epochs = argu.epochs
capacity = argu.capacity
#noDevCriterion = argu.noDevCriterion
eta =0.1

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
            FeatureArrayPlus[j, 0] = 1
    return FeatureArrayPlus, Label

def Train(FeatureArray, Label, eta, epochs, capacity):
    weights = np.zeros(len(FeatureArray[0,:]))
    for i in range(epochs):
        for j in range(0, len(FeatureArray[:,0])):
            DotProduct = np.dot(weights, FeatureArray[j,:])
            if DotProduct*Label[j] <= 1  :
                weights[1:] = weights[1:] - 1/len(FeatureArray[:,1])*eta*weights[1:] + capacity*eta*Label[j]*FeatureArray[j,1:]
                weights[0] = weights[0]+ capacity*eta*Label[j]*FeatureArray[j,0]
            else:
                weights[1:] = weights[1:] - 1/len(FeatureArray[:,1])*eta*weights[1:]
    return weights   
    
def Test(FeatureArraytest, Labeltest, eta):
    Count = 0 
    right = 0
    for j in range(0, len(FeatureArraytest[:,0])):
        DotProduct = np.dot( weights, FeatureArraytest[j,:])
        sign = np.sign(DotProduct)
        Count=Count+1
        if sign * Labeltest[j] > 0 :
            right=right+1
    Accuracy = right/Count
    return Accuracy

#FeatureArray, Label = ReadData('/u/cs246/data/adult/a7a.train')  ### Reading the traning data 
FeatureArray, Label = ReadData('a7a.train')  ### Reading the traning data

weights = Train (FeatureArray, Label, eta, epochs, capacity)  ### Traning using perceptron 
        
#FeatureArraytest, Labeltest = ReadData('/u/cs246/data/adult/a7a.test')  ### Reading the test data
FeatureArrayTrain, LabelTrain = ReadData('a7a.train')  ### Reading the test data
AccuracyTrain = Test(FeatureArrayTrain, LabelTrain, eta)
  
#FeatureArraytest, Labeltest = ReadData('/u/cs246/data/adult/a7a.test')  ### Reading the test data
FeatureArrayTest, LabelTest = ReadData('a7a.test')  ### Reading the test data
AccuracyTest = Test(FeatureArrayTest, LabelTest, eta)

#FeatureArraytest, Labeltest = ReadData('/u/cs246/data/adult/a7a.test')  ### Reading the test data
FeatureArrayDev, LabelDev = ReadData('a7a.dev')  ### Reading the test data
AccuracyDev = Test(FeatureArrayDev, LabelDev, eta)

print('EPOCHS:', epochs)
print('CAPACITY:', capacity)
print('TRAINING_ACCURACY:', AccuracyTrain)
print('TEST_ACCURACY :', AccuracyTest) 
print('DEV_ACCURACY', AccuracyDev)
print('FINAL_SVM: ',  *weights) 
