'''
Created on Aug 29, 2017

@author: raul
'''
import numpy as np
from Base import Base
from funcoesAuxiliares import separarElementosPorClasse
import math
import copy

def pca (base1,k=0):
    base = Base(copy.deepcopy(base1.classes),copy.deepcopy(base1.atributos))
    media = np.mean(np.array(base.atributos),axis=0)
    subtracao = np.array(copy.deepcopy(base.atributos))
    #cria a matriz de subtração
    for j in range(len(base.atributos[0])):
        for i in range(len(base.atributos)):
            subtracao[i][j] = subtracao[i][j] - media[j]
    
    cov = (1/len(media))*np.dot(subtracao.T,subtracao)
    autoValues,autoVectors = np.linalg.eig(cov)
    autoValues,autoVectors = zip(*sorted(zip(autoValues, autoVectors ),reverse=True))
    autoVectors = autoVectors[0:len(base.atributos[0])-k]
    autoValues = autoValues[0:len(base.atributos[0])-k]
    novosAtributos = np.dot(subtracao,np.array(autoVectors).T)
    return (Base(base.classes,novosAtributos))

def pcaScore(base1,k=0):
    base = Base(copy.deepcopy(base1.classes),copy.deepcopy(base1.atributos))
    media = np.mean(np.array(base.atributos),axis=0)
    subtracao = np.array(copy.deepcopy(base.atributos))
    #cria a matriz de subtração
    for j in range(len(base.atributos[0])):
        for i in range(len(base.atributos)):
            subtracao[i][j] = subtracao[i][j] - media[j]
    
    cov = (1/len(media))*np.dot(subtracao.T,subtracao)
    autoValues,autoVectors = np.linalg.eig(cov)
    m1,m2 = separarElementosPorClasse(base,base.classes)
    m1 = np.mean(m1, axis=0)
    m2 = np.mean(m2, axis=0)
    scores = score(m1, m2, autoValues)
    scores,autoVectors = zip(*sorted(zip(scores, autoVectors ),reverse=True))
    autoVectors = autoVectors[0:len(base.atributos[0])-k]
    novosAtributos = np.dot(subtracao,np.array(autoVectors).T)
    return (Base(base.classes,novosAtributos))

def score(media1,media2,autovalores):
    scores = []
    for i in range(len(media1)):
        s = 0
        if(autovalores[i] != 0):
            s = math.sqrt(math.pow((media1[i]-media2[i]),2))/autovalores[i]
        scores.append(s)
    return scores
    
    