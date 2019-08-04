# -*- coding: utf-8 -*-


from __future__ import division
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np
import csv
import os

files = os.listdir('testing_data')

total_classes = len(files)

actual = [] 
predicted = []
class_accuracy = []
avg = 0.0

for filename in files:
    
    T = 0
    F = 0
    i = 0
    
    acc = []
    pred = []
    
    #print('testing_data/'+ filename +'.csv')
    with open('testing_data/'+ filename , 'rU') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            break
        for row in spamreader:
            
            actual.append(row[1])            
            predicted.append(row[2])
            acc.append(row[1])
            pred.append(row[2])
        
        class_accuracy.append(accuracy_score(acc, pred))

y_true = actual
y_pred = predicted
confusion_matrix = confusion_matrix(y_true, y_pred, labels=files)
print
print
for i in range(0, total_classes):
    
    TP = confusion_matrix[i, i] 
    FP = np.sum(confusion_matrix, axis=0)[i] - confusion_matrix[i, i]  #The corresponding column for class_i - TP
    FN = np.sum(confusion_matrix, axis=1)[i] - confusion_matrix[i, i] # The corresponding row for class_i - TP
    TN = np.sum(confusion_matrix) - TP -FP -FN
    
    accuracy = (TP + TN)/(TP + TN + FP + FN)

    print ("class: ", files[i]   )
    print ('accuracy: ', class_accuracy[i]*100, '%')
    print ('True Pos: ',TP,'True Neg: ', TN)
    print ('False Pos: ', FP, 'False Neg: ', FN)
    print

print ('Confusion matrix: ')
print  (confusion_matrix)
print 
print ('Accuracy: ', accuracy_score(y_true, y_pred)*100, '%')
print
print ('Report: ')
print (classification_report(y_true, y_pred))
