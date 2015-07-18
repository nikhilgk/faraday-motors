#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import *
import csv

#import the merged charging station and ACS data
lines = open('add_0_stations.csv').read().splitlines()
#Remove the first row, it's just the headings
lines.pop(0)

zip_code, count, population, male_pct, age, college_pct, income, home_pct = [], [], [], [], [], [], [], []
for line in lines:
    items = line.split(',')
    zip_code.append(items[0])
    count.append(items[1])
    population.append(items[2])
    male_pct.append(items[3])
    age.append(items[4])
    college_pct.append(items[5])
    income.append(items[6])
    home_pct.append(items[7])
    
    
# add shuffle code before splitting the data into train and dev sets

# Split the imported data into training 75% and development data 25%

split = 3*len(zip_code)/4

print len(zip_code), split

trainZipCode, trainCount, trainPopulation, trainMalePercentage, trainAge, trainCollegePercentage, trainIncome, trainHomeOwnerPercentage \
= zip_code[:split], count[:split], population[:split], male_pct[:split], age[:split], college_pct[:split], income[:split], home_pct[:split]

devZipCode, devCount, devPopulation, devMalePercentage, devAge, devCollegePercentage, devIncome, devHomeOwnerPercentage \
= zip_code[split:], count[split:], population[split:], male_pct[split:], age[split:], college_pct[split:], income[split:], home_pct[split:]

print len(trainZipCode)
train_labels = np.array(trainCount)
print train_labels.shape
X1 = np.array(trainPopulation)
X2 = np.array(trainMalePercentage)
train_data = np.vstack((X1,X2,trainAge,trainCollegePercentage,trainIncome,trainHomeOwnerPercentage))
train_data = np.transpose(train_data)
print X1.shape, X2.shape, train_data.shape
dev_data=np.vstack((devPopulation, devMalePercentage, devAge, devCollegePercentage, devIncome, devHomeOwnerPercentage))
dev_data = np.transpose(dev_data)
dev_labels=np.array(devCount)
print dev_data.shape, dev_labels.shape
print train_data[0], train_labels[0]
print dev_data[0], dev_labels[0]

# fit the first model with logistic regression 
#
logreg = LogisticRegression(penalty='l1', C=1.0, tol=0.01) 
logreg.fit(train_data, train_labels)
print logreg.score(dev_data, dev_labels)


