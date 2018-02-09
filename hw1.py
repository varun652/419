
# coding: utf-8

# In[4]:


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


# In[303]:


import numpy as np
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


# In[304]:


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


# In[430]:


from sklearn.model_selection import cross_val_score
from sklearn import metrics

#SGD with tweaked parameters
clf = SGDClassifier(loss="log", tol = 1, learning_rate = 'constant', eta0 = 0.1)

#5-fold CV
SGDScores = cross_val_score(clf, xVals, yVals, cv=5)
print(SGDScores)
print('Average:',sum(SGDScores)/len(SGDScores))

# Accuracy on training data
clf.fit(xVals,yVals)
yPredict = clf.predict(xVals)
SGDAccuracy = accuracy_score(yVals, yPredict)
print('Training Accuracy',SGDAccuracy)


# In[504]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree

#DT with no Max Depth
clf_dt = DecisionTreeClassifier()
DTScores = cross_val_score(clf_dt, xVals, yVals, cv=5)
print(DTScores)
print('Average:',sum(DTScores)/len(DTScores))    

#DT with Max Depth 4
clf_dt1 = DecisionTreeClassifier(max_depth=4)
DT1Scores = cross_val_score(clf_dt1, xVals, yVals, cv=5)
print(DT1Scores)
print('Average:',sum(DT1Scores)/len(DT1Scores))   

#DT with Max Depth 8
clf_dt2 = DecisionTreeClassifier(max_depth=8)
DT2Scores = cross_val_score(clf_dt2, xVals, yVals, cv=5)
print(DT2Scores)
print('Average:',sum(DT2Scores)/len(DT2Scores))    


# In[553]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import random
#Creating subsets to train DT stumps
xStumpValues = np.zeros((350,260))
yStumpValues = np.zeros(350)
#List for storing DTs
dtList = []
#Selecting 50% of the sample
#Obtaining Training Data for each stump
for i in range(100):
    indices = random.sample(range(0, 700), 350)
    for j in range(len(indices)):
        yStumpValues[j] = yVals[indices[j]]
        for k in range(260):
            xStumpValues[j][k] = xVals[indices[j]][k]
    clf_dstump = DecisionTreeClassifier(max_depth=8)
    dtList.append(clf_dstump.fit(xStumpValues,yStumpValues))


# In[554]:


#Predictions for Features from stumps
xStumpFeatures = np.zeros((len(xVals),100))
for i in range(100):
    xStumpFeatures[:,i] = dtList[i].predict(xVals)


# In[585]:


clf = SGDClassifier(loss="log", tol = 1, learning_rate = 'constant', eta0 = 0.1)
SGDScores = cross_val_score(clf, xStumpFeatures, yVals, cv=5)
print(SGDScores)
print('Average:', sum(SGDScores)/len(SGDScores))
clf.fit(xStumpFeatures,yVals)
yStumpVals = clf.predict(xStumpFeatures)
SGDAccuracy = accuracy_score(yVals, yStumpVals)
print('Training Accuracy',SGDAccuracy)

