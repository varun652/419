
# coding: utf-8

# In[798]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.tree import export_graphviz
import pandas as pd
import random
import numpy as np

#Loading Data
dataTraining=[]
with open('badges.modified.data.train') as myfile:
    for line in myfile:
        dataTraining.append(line.rstrip())
data1=[]
with open('badges.modified.data.fold1') as myfile:
    for line in myfile:
        data1.append(line.rstrip())
data2=[]
with open('badges.modified.data.fold2') as myfile:
    for line in myfile:
        data2.append(line.rstrip())
data3=[]
with open('badges.modified.data.fold3') as myfile:
    for line in myfile:
        data3.append(line.rstrip())
data4=[]
with open('badges.modified.data.fold4') as myfile:
    for line in myfile:
        data4.append(line.rstrip())
data5=[]
with open('badges.modified.data.fold5') as myfile:
    for line in myfile:
        data5.append(line.rstrip())
dataTest=[]
with open('badges.modified.data.test') as myfile:
    for line in myfile:
        dataTest.append(line.rstrip())


# In[799]:


#Extract Labels
def yFeatures (array):
    yVals = np.zeros(len(array))
    i = 0
    for x in array:
        myString = x
        myList = myString.split(" ")
        if myList[0] == '-':
             yVals[i] = 0
        else:
             yVals[i] = 1
        i = i + 1
    return yVals;
yVals = yFeatures(dataTraining)


# In[800]:


#Extract Features
def xFeatures (array):
    xVals = np.zeros((len(array),260)) 
    posVertical = 0;
    for x in array:
        myString = x
        myList = myString.split(" ")
        for i in range(len(myList[1])):
            if(i < 5):
                xVals[posVertical][26 * i + ord(myList[1][i]) - 97] = 1
        for j in range(len(myList[2])):
            if(j < 5):
                xVals[posVertical][26 * (5 + j) + ord(myList[2][j]) - 97] = 1
        posVertical = posVertical + 1;
    return xVals;
xVals = xFeatures(dataTraining)


# In[801]:


#SGD with tweaked parameters
clf = SGDClassifier(loss="log", tol = 0.005, learning_rate = 'constant', eta0 = 0.1)

#5-fold CV
SGDScores = cross_val_score(clf, xVals, yVals, cv=5)
print(SGDScores)
print('Average:',sum(SGDScores)/len(SGDScores))

# Accuracy on training data
clf.fit(xVals,yVals)
yPredict = clf.predict(xVals)
SGDAccuracy = accuracy_score(yVals, yPredict)
print('Training Accuracy',SGDAccuracy)


# In[802]:


#DT with no Max Depth
clf_dt = DecisionTreeClassifier()
DTScores = cross_val_score(clf_dt, xVals, yVals, cv=5)
print(DTScores)
print('Average:',sum(DTScores)/len(DTScores))    

# Accuracy on training data
clf_dt.fit(xVals,yVals)
yPredict = clf_dt.predict(xVals)
DTAccuracy = accuracy_score(yVals, yPredict)
print('Training Accuracy',DTAccuracy)

#DT with Max Depth 4
clf_dt1 = DecisionTreeClassifier(max_depth=4)
DT1Scores = cross_val_score(clf_dt1, xVals, yVals, cv=5)
print(DT1Scores)
print('Average:',sum(DT1Scores)/len(DT1Scores))   

# Accuracy on training data
clf_dt1.fit(xVals,yVals)
yPredict = clf_dt1.predict(xVals)
DT1Accuracy = accuracy_score(yVals, yPredict)
print('Training Accuracy',DT1Accuracy)

#DT with Max Depth 8
clf_dt2 = DecisionTreeClassifier(max_depth=8)
DT2Scores = cross_val_score(clf_dt2, xVals, yVals, cv=5)
print(DT2Scores)
print('Average:',sum(DT2Scores)/len(DT2Scores))    

# Accuracy on training data
clf_dt2.fit(xVals,yVals)
yPredict = clf_dt2.predict(xVals)
DT2Accuracy = accuracy_score(yVals, yPredict)
print('Training Accuracy',DT2Accuracy)


# In[803]:


#Creating subsets to train DT stumps
xStumpValues = np.zeros((350,260))
yStumpValues = np.zeros(350)
#List for storing DTs
dtList4 = []
dtList8 = []
#Selecting 50% of the sample
#Obtaining Training Data for each stump
for i in range(100):
    indices = random.sample(range(0, 700), 350)
    for j in range(len(indices)):
        yStumpValues[j] = yVals[indices[j]]
        for k in range(260):
            xStumpValues[j][k] = xVals[indices[j]][k]
    clf_dstump4 = DecisionTreeClassifier(max_depth=4)
    clf_dstump8 = DecisionTreeClassifier(max_depth=8)
    dtList4.append(clf_dstump4.fit(xStumpValues,yStumpValues))
    dtList8.append(clf_dstump8.fit(xStumpValues,yStumpValues))


# In[804]:


#Predictions for Features from stumps
xStumpFeatures4 = np.zeros((len(xVals),100))
xStumpFeatures8 = np.zeros((len(xVals),100))
for i in range(100):
    xStumpFeatures4[:,i] = dtList4[i].predict(xVals)
    xStumpFeatures8[:,i] = dtList8[i].predict(xVals)


# In[805]:


clf = SGDClassifier(loss="log", learning_rate = 'constant', eta0 = 0.1)
#SGD Classifier for stump depth 8
SGDScores = cross_val_score(clf, xStumpFeatures8, yVals, cv=5)
print(SGDScores)
print('Average:', sum(SGDScores)/len(SGDScores))
clf.fit(xStumpFeatures8,yVals)
yStumpVals = clf.predict(xStumpFeatures8)
SGDAccuracy = accuracy_score(yVals, yStumpVals)
print('Training Accuracy',SGDAccuracy)

#SGD Classifier for stump depth 4
SGDScores = cross_val_score(clf, xStumpFeatures4, yVals, cv=5)
print(SGDScores)
print('Average:', sum(SGDScores)/len(SGDScores))
clf.fit(xStumpFeatures4,yVals)
yStumpVals = clf.predict(xStumpFeatures4)
SGDAccuracy = accuracy_score(yVals, yStumpVals)
print('Training Accuracy',SGDAccuracy)


# In[806]:


#Working with the test data
xTestLabels = np.zeros((len(dataTest),260)) 
posVertical = 0;
for x in dataTest:
    myString = x
    myList = myString.split(" ")
    for i in range(len(myList[0])):
        if(i < 5):
            xTestLabels[posVertical][26 * i + ord(myList[0][i]) - 97] = 1
    for j in range(len(myList[1])):
        if(j < 5):
            xTestLabels[posVertical][26 * (5 + j) + ord(myList[1][j]) - 97] = 1
    posVertical = posVertical + 1;

#Creating the new features for SGD Decision Stump method
xNewFeatures = np.zeros((len(xTestLabels),100))
for i in range(100):
    xNewFeatures[:,i] = dtList8[i].predict(xTestLabels)
#Prediction of Test Labels
yTestLabels = clf.predict(xNewFeatures)
yFinalLabels = []
print(yTestLabels)
for x in range(len(yTestLabels)):
    if(yTestLabels[x] == 0):
        yFinalLabels.append('-')
    else: yFinalLabels.append('+')

print(yFinalLabels)


# In[807]:


#Printing labels to file
f = open("test_labels.txt", "w")
for i in range(len(yFinalLabels)):
    f.write(str(yFinalLabels[i]) + ' '  + str(dataTest[i]) + '\n')
f.close() 


# In[808]:


#Decision tree image creator
export_graphviz(clf_dt, 
                filled=True, rounded=True,
                special_characters=True)

export_graphviz(clf_dt1, 
                filled=True, rounded=True,
                special_characters=True)

export_graphviz(clf_dt2, 
                filled=True, rounded=True,
                special_characters=True)

