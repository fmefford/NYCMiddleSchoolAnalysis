# -*- coding: utf-8 -*-
"""
Created on Mon May 17 20:43:00 2021

@author: Finn Mefford
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas
from sklearn import linear_model
from sklearn.decomposition import PCA
import scipy.stats as stats

pdData = pandas.read_csv(r'C:\Users\finnm\Desktop\DataScienceFinals\middleSchoolData.csv',
                       sep = ",", header = 0, index_col = [0, 1])
data = pdData.to_numpy()

def nanHandler(inData, colsToCheck):
    outData = np.copy(inData[:, colsToCheck])
    whereAreTheNansAt = np.isnan(outData)
    nanCoordinates = np.where(whereAreTheNansAt==True)
    return np.delete(outData, nanCoordinates[0], 0)

def getSchool(row):
    return pdData.index[row]

def pcaEigensum(eigVals):
    totalVariance = 0
    eigSumAt90 = 0
    for val in eigVals:
        totalVariance += val
    for i in range(len(eigVals)):
        eigSumAt90 += eigVals[i]
        if (eigSumAt90 / totalVariance) >= .9:
            return i, eigSumAt90 / totalVariance
        
def acceptanceRateCalc(acceptancesAndPopulation):
    acceptanceRates = np.copy(acceptancesAndPopulation[:, 0])
    for i in range(acceptancesAndPopulation.shape[0]):
        acceptanceRates[i] /= acceptancesAndPopulation[i, 1]
    return acceptanceRates
    
#Question 1
data1 = nanHandler(data, [0, 1])
r1 = np.corrcoef(data1[:, 0], data1[:, 1])
#print("Question 1: ", r1[1, 0])

#Question 2
data2 = nanHandler(data, [0, 1, 18])
for i in range(data2.shape[0]):
    data2[i, 0] /= data2[i, 2]
r2 = np.corrcoef(data2[:, 0], data2[:, 1])
#print("Question 2: ", r2[1, 0])

#Question 3
data3 = nanHandler(data, [1, 18])
acceptanceRates3 = acceptanceRateCalc(data3)

highestPerStudent = np.argmax(acceptanceRates3)
#print("Question 3:", getSchool(highestPerStudent)[1], ", ", acceptanceRates3[highestPerStudent])

#Question 4
data4 = nanHandler(data, [9, 10, 11, 12, 13, 14, 19, 20, 21])
data5 = np.copy(data4[:, 6:])
data4 = np.copy(data4[:, :6])

zScoredData4 = stats.zscore(data4)
pca4 = PCA()
pca4.fit(zScoredData4)
eigVals4 = pca4.explained_variance_
loadings4 = pca4.components_
rotatedData4 = pca4.fit_transform(zScoredData4)
numOfPCs4 = pcaEigensum(eigVals4)[0]

zScoredData5 = stats.zscore(data5)
pca5 = PCA()
pca5.fit(zScoredData5)
eigVals5 = pca5.explained_variance_
loadings5 = pca5.components_
rotatedData5 = pca5.fit_transform(zScoredData5)
numOfPCs5 = pcaEigensum(eigVals5)[0]

"""
for i in range(numOfPCs4 + 1):
    plt.figure()
    plt.title("Perception Loadings {}".format(i + 1))
    plt.bar([1, 2, 3, 4, 5, 6],loadings4[:, i])
    
for j in range(numOfPCs5 + 1):
    plt.figure()
    plt.title("Achievement Loadings {}".format(j + 1))
    plt.bar([1, 2, 3],loadings5[:, j])
"""
    
for i in range(numOfPCs4 + 1):
    for j in range(numOfPCs5 + 1):
        break
        #print("Perception", i + 1, ", Achievement", j + 1, ": ", np.corrcoef(rotatedData4[:, i], rotatedData5[:, j])[1, 0])

#Question 5
#How do racialy homogeneous schools perform on hsphs admissions compared to racially diverse
#A racially homogenous school's racial majority account for a percent of the student
#body greater than or equal to the median 

data6 = nanHandler(data, [1, 4, 5, 6, 7, 8, 18])
acceptanceRates6 = acceptanceRateCalc(data6[:, [0, 6]])

maxRace = np.empty(data6.shape[0])
for i in range(data6.shape[0]):
    maxRace[i] = data6[i, [1, 2, 3, 4, 5]].max()
maxRaceMedian = np.median(maxRace)
for i in range(maxRace.shape[0]):
    if maxRace[i] >= maxRaceMedian:
        maxRace[i] = 1
    else:
        maxRace[i] = 0
sortedRace = np.column_stack([acceptanceRates6, maxRace])
sortedRace = sortedRace[sortedRace[:, 1].argsort()]
val = sortedRace[0, 1]
pos = 0
while val == 0:
    pos += 1
    val = sortedRace[pos, 1]
homogeneous = sortedRace[pos:, 0]
diverse = sortedRace[:pos, 0]
#print(stats.levene(homogeneous, diverse))
#print(stats.mannwhitneyu(diverse, homogeneous))
#print(stats.pointbiserialr(sortedRace[:, 1], sortedRace[:, 0]))

#Question 6
data7 = nanHandler(data, [2, 3, 19, 20, 21])
data8 = np.copy(data7[:, 2:])
data7 = np.copy(data7[:, :2])
plt.figure()
plt.scatter(data7[:, 0], data8[:, 0])
plt.scatter(data7[:, 0], data8[:, 1])
plt.scatter(data7[:, 0], data8[:, 2])
plt.xlabel('per student spending ')
plt.ylabel('achievement')
plt.show()
plt.figure()
plt.scatter(data7[:, 1], data8[:, 0])
plt.scatter(data7[:, 1], data8[:, 1])
plt.scatter(data7[:, 1], data8[:, 2])
plt.xlabel('class size')
plt.ylabel('achievement')
plt.show()
"""
for i in range(data7.shape[1]):
    if i == 0:
        print("Spending")
    else:
        print("Size")
    for j in range(data8.shape[1]):
        if j == 0:
            print("Student Achievement", end= ", ")
        elif j == 1:
            print("Reading Scores", end= ", ")
        else:
            print("Math Scores", end= ", ")
        print(np.corrcoef(data7[:, i], data8[:, j])[1, 0])
"""
#Question 7
data9 = nanHandler(data, [1])

data9 = np.sort(np.transpose(data9))

totalAcceptances = 0
for i in range(data9.shape[1]):
    totalAcceptances += data9[0, i]

runningTotal = 0
index = data9.shape[1] - 1
while (runningTotal < totalAcceptances * .9):
    runningTotal += data9[0, index]
    index -= 1

numOfSchools = data9.shape[1] - index
#print(numOfSchools / data9.shape[1])

"""
plt.figure()
plt.bar(np.arange(numOfSchools), data9[0, index: data9.shape[1]])
plt.xlabel('School')
plt.ylabel('Acceptances')
"""

#Question 8
data10 = nanHandler(data, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
regressors = np.copy(data10[:, [0, 18, 19, 20]])
for i in range(regressors.shape[0]):
    regressors[i, 0] /= data10[i, 17]
predictors = np.copy(data10[:, 1: 18])
"""
regr = linear_model.LinearRegression()
regr.fit(predictors, regressors[:, 0]) # use fit method 
betas = regr.coef_ 
print(betas)
yInt = regr.intercept_
for i in range(predictors.shape[0]):
    print(regr.predict(predictors[i, :].reshape(1, -1)))
"""
regrAcceptances = linear_model.Lasso(alpha = .01)
regrAcceptances.fit(predictors, regressors[:, 0])
betas = regrAcceptances.coef_
y_int = regrAcceptances.intercept_
y_hat = 0
for i in range(len(betas)):
    y_hat += betas[i] * predictors[:, i]
    
y_hat += y_int
"""
for i in range(predictors.shape[0]):
    print(regrAcceptances.predict(predictors[i, :].reshape(1, -1)))
"""
   
regrAchievement = linear_model.Lasso(alpha = .01)
regrAchievement.fit(predictors, regressors[:, 1])
betas = regrAchievement.coef_
y_int = regrAchievement.intercept_
y_hat = 0
for i in range(len(betas)):
    y_hat += betas[i] * predictors[:, i]
    
y_hat += y_int
"""
for i in range(predictors.shape[0]):
    print(regrAchievement.predict(predictors[i, :].reshape(1, -1)))
"""
    
regrReading = linear_model.Lasso(alpha = .01)
regrReading.fit(predictors, regressors[:, 2])
betas = regrReading.coef_
y_int = regrReading.intercept_
y_hat = 0
for i in range(len(betas)):
    y_hat += betas[i] * predictors[:, i]
    
y_hat += y_int
"""
for i in range(predictors.shape[0]):
    print(regrReading.predict(predictors[i, :].reshape(1, -1)))
"""
    
regrMath = linear_model.Lasso(alpha = .01)
regrMath.fit(predictors, regressors[:, 3])
betas = regrMath.coef_
y_int = regrMath.intercept_
y_hat = 0
for i in range(len(betas)):
    y_hat += betas[i] * predictors[:, i]
    
y_hat += y_int
"""
for i in range(predictors.shape[0]):
    print(regrMath.predict(predictors[i, :].reshape(1, -1)))
"""
    
regr = linear_model.Lasso(alpha = .25)
regr.fit(predictors, regressors)
betas = regr.coef_
y_int = regr.intercept_
y_hat = 0
#print(betas)
#print(regr.score(predictors, regressors))
#for i in range(len(betas)):
    #y_hat += betas[i] * predictors[:, i]
  
"""
data12 = nanHandler(data, [1, 2])
plt.figure()
plt.scatter(data12[:, 1], data12[:, 0])
plt.xlabel('spending per student')
plt.ylabel('poverty')
plt.show()
"""


#print(np.corrcoef(data12[:, 0], data12[:, 1]))

#Are rich schools actually worse
data11 = nanHandler(data, [2, 19])
median = np.median(data11[:, 0])
for i in range(data11.shape[0]):
    if data11[i, 0] >= median:
        data11[i, 0] = 1
    else:
        data11[i, 0] = 0
        
sortedRich = data11[data11[:, 0].argsort()]

val = sortedRich[0, 0]
pos = 0
while val == 0:
    pos += 1
    val = sortedRace[pos, 0]
poor = sortedRich[pos:, 1]
rich = sortedRich[:pos, 1]
#print(np.mean(rich - poor))


"""
zScoredData7 = stats.zscore(data7)
pca7 = PCA()
pca7.fit(zScoredData7)
eigVals7 = pca7.explained_variance_
loadings7 = pca7.components_
rotatedData7 = pca7.fit_transform(zScoredData7)
numOfPCs7 = pcaEigensum(eigVals7)[0]

zScoredData8 = stats.zscore(data8)
pca8 = PCA()
pca8.fit(zScoredData8)
eigVals8 = pca8.explained_variance_
loadings8 = pca8.components_
rotatedData8 = pca8.fit_transform(zScoredData8)
numOfPCs8 = pcaEigensum(eigVals8)[0]

plt.figure()
plt.bar([0, 1], eigVals7)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([-.5,6],[1,1],color='red',linewidth=1) 

plt.figure()
plt.bar([0, 1, 2], eigVals8)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([-.5,6],[1,1],color='red',linewidth=1) 

for i in range(numOfPCs7 + 1):
    plt.figure()
    plt.title("Perception Loadings {}".format(i + 1))
    plt.bar([1, 2],loadings7[:, i])
    
for j in range(numOfPCs8 + 1):
    plt.figure()
    plt.title("Achievement Loadings {}".format(j + 1))
    plt.bar([1, 2, 3],loadings8[:, j])
    
for i in range(numOfPCs7 + 1):
    for j in range(numOfPCs8 + 1):
        print("Resources", i + 1, ", Achievement", j + 1, ": ", np.corrcoef(rotatedData4[:, i], rotatedData5[:, j])[1, 0])
"""

"""
plt.figure()
plt.bar([0, 1, 2, 3, 4, 5], eigVals4)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([-.5,6],[1,1],color='red',linewidth=1) 

plt.figure()
plt.bar([1, 2, 3, 4, 5, 6],loadings4[:,0])
plt.figure()
plt.bar([1, 2, 3, 4, 5, 6],loadings4[:,1])
plt.figure()
plt.bar([1, 2, 3, 4, 5, 6],loadings4[:,2])
plt.figure()
plt.bar([1, 2, 3, 4, 5, 6],loadings4[:,3])
plt.xlabel('Question')
plt.ylabel('Loading')

plt.figure()
plt.bar([0, 1, 2], eigVals5)
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([-.5,6],[1,1],color='red',linewidth=1) 

plt.figure()
plt.scatter(data2[:, 0], data2[:, 1])
plt.xlabel('applications')
plt.ylabel('acceptances')
plt.show()
"""