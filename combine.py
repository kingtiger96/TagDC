# coding=gbk
from keras.callbacks import EarlyStopping
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop
import keras.backend.tensorflow_backend as K
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Model
from sklearn.metrics import  precision_score ,recall_score
import heapq
from sklearn.metrics import accuracy_score
import tensorflow as tf
import os
from sklearn.metrics import classification_report
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau
from keras import backend as K1
from keras.layers import Layer
from keras import initializers, regularizers, constraints

import pandas


def topk(array_list,i):
   
    max_num_index_list = map(array_list.index, heapq.nlargest(i, array_list))
    return (list(max_num_index_list))


def cfscorecal():
    labels = []
    for i in range(1,4785):
        labs = open("/data/lican/StackOverflowsmall/cf/Tag_numpy%ld.txt"%(i)).read().split()   
        labels.append(labs)
    return labels
    

def cfProcess():
    labels = cfscorecal()
    labs = []
    for label in labels:
        label = list(map(float,label))
        labs.append(label)
    return labs    



def dlscorecal():
    labels = []
    for i in range(1,4785):
        labs = open("/data/lican/StackOverflowsmall/dl/Tag_numpy%ld.txt"%(i)).read().split()   
        labels.append(labs)
    return labels
    

def dlProcess():
    labels = dlscorecal()
    labs = []
    for label in labels:
        label = list(map(float,label))
        labs.append(label)
    return labs    




def testscorecal():
    labels = []
    for i in range(1,4785):
        labs = open("/data/lican/StackOverflowsmall/test/Tag_numpy%ld.txt"%(i)).read().split()   
        labels.append(labs)
    return labels
    

def testProcess():
    labels = testscorecal()
    labs = []
    for label in labels:
        label = list(map(int,label))
        labs.append(label)
    return labs    


def combine(k,cfscore,dlscore):
    finalscore = []
    for i in range(len(cfscore)):
        score = (k*cfscore[i] + dlscore[i])/(1+k)
        finalscore.append(score)
    return finalscore


cfscore = cfProcess()
cfscore = np.array(cfscore)
dlscore = dlProcess()
dlscore = np.array(dlscore)
testscore = testProcess()
testscore = np.array(testscore)


finalscore = combine(0.069,cfscore,dlscore)

   
y_prediction5 = []
    
for i in range(len(finalscore)):
    yy = []
    max_index = topk(finalscore[i].tolist(),5)
    for p in range(len(finalscore[i].tolist())):
        if p in max_index:
            yy.append(1)
        else:
            yy.append(0)
    y_prediction5.append(yy)
    yy = []
y_true = np.vstack(testscore)
y_pred5 = np.vstack(y_prediction5)

y_prediction10 = []
    
for i in range(len(finalscore)):
    yy = []
    max_index = topk(finalscore[i].tolist(),10)
    for p in range(len(finalscore[i].tolist())):
        if p in max_index:
            yy.append(1)
        else:
            yy.append(0)
    y_prediction10.append(yy)
    yy = []
y_pred10 = np.vstack(y_prediction10)
tags = open('/data/lican/StackOverflowsmall/Tags_list.txt').read().split()
print("top5")
#print("top5----precision_score:",precision_score(y_true, y_pred5,average = "samples"))
print("top5----recall_score:",recall_score(y_true, y_pred5,average = 'samples'))
print(classification_report(y_true, y_pred5,target_names=tags)) 
"""
report1 = classification_report(y_true, y_pred5,output_dict=True)
df = pandas.DataFrame(report1).transpose()
df.to_excel('/data/lican/StackOverflowsmall/final5.xlsx')


    
print("top10")
#print("top10----precision_score:",precision_score(y_true, y_pred10,average = "samples"))
print("top10----recall_score:",recall_score(y_true, y_pred10,average = 'samples'))
print(classification_report(y_true, y_pred10))

report2 = classification_report(y_true, y_pred10,output_dict=True)
df = pandas.DataFrame(report2).transpose()
df.to_excel('/data/lican/StackOverflowsmall/final10.xlsx')
"""