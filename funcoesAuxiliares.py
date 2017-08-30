'''
Created on Aug 29, 2017

@author: raul
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from numba import jit
import numpy as np
import copy
from Base import Base
from sklearn.metrics import accuracy_score
#Classificador knn para atributos numéricos
def classicarKNN(base,n=1):
    knn = KNeighborsClassifier(n_neighbors=n)
    erroKnn = 0
    loo = LeaveOneOut()
    X = np.array(base.atributos)
    y = np.array(base.classes)
    classesAchadas = []
    classeCerta = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn.fit(X_train,y_train)
        knnPredict = knn.predict(X_test)
        erroKnn = (1-accuracy_score(y_test,knnPredict)) + erroKnn
        classesAchadas.append(knnPredict[0])
        classeCerta.append(y_test[0])
        '''
        for i,e in enumerate(knnPredict):
            
            if(e!=y_test[i]):
                erroKnn = erroKnn + 1
            classesAchadas.append(e)
        '''
    return erroKnn/len(base.atributos),classesAchadas,classeCerta

#Função responsável por discretizar os valores númericos
#de uma base
@jit
def discretizacao(base,intervalo=2):
    colunas = np.array(base.atributos).T
    dic = {}
    inter = []
    c = copy.deepcopy(base.classes)
    a = copy.deepcopy(base.atributos)
    baseDiscretizada = Base(c,a)
    for i,atr in enumerate(colunas):
        dic[i] = [max(atr),min(atr)]
    for e in dic:
        inter.append(float((dic[e][0]-dic[e][1])/(intervalo)))
    for j in range(len(baseDiscretizada.atributos[0])):
        for i in range(len(baseDiscretizada.atributos)):
            baseDiscretizada.atributos[i][j] =  int((baseDiscretizada.atributos[i][j] - dic[j][1])/inter[j])
    return baseDiscretizada

@jit
def separarElementosPorClasse(base,classes):
    m1 = []
    m2 = []
    for i,e in enumerate(base.classes):
        if(e==classes[0]):
            m1.append(base.atributos[i])
        else:
            m2.append(base.atributos[i])
    return m1,m2

@jit
def separarElementosPorClasse2(base,classes):
    m1 = []
    m2 = []
    c1 = []
    c2 = []
    for i,e in enumerate(base.classes):
        if(e==classes[0]):
            m1.append(base.atributos[i])
            c1.append(classes[0])
        else:
            m2.append(base.atributos[i])
            c2.append(classes[1])
    return Base(c1,m1),Base(c2,m2) 
