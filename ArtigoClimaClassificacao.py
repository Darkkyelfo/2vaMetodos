'''
Created on Aug 30, 2017

@author: raul1
'''

from funcoesAuxiliares import classicarKNN,separarElementosPorClasse
from NaiveBayes import NaiveBayes
from Base import Base
from PCA import pcaScore,pca
import numpy as np
def separaBaseClima():
    arq = open("BasesArtigo/clima")
    linhas = arq.readlines()
    atributos = []
    classes = []
    novaLista = []
    for i in linhas[1:]:
        temp = []
        a = i.split(" ")
        for e in a:
            if(e != "" and e!="\n"):
                temp.append(float(e))
        novaLista.append(temp)
    arq.close()
    for e in novaLista:
        classes.append(e[len(e)-1])
        atributos.append(e[2:len(e)-1])
    return Base(classes,atributos)

baseClima = separaBaseClima()
p = [0] + list(range(1,17,2))

for k in p:
    baseClimePCA = pcaScore(baseClima, k)
    m1,m2 = separarElementosPorClasse(baseClimePCA, baseClimePCA.classes)
    v1 = np.cov(np.array(m1).T)
    v2 = np.cov(np.array(m2).T)
    m1 = np.mean(m1, axis=0)
    m2 = np.mean(m2, axis=0)
    print("acerto com k=%s - 1-nn:%s" %(k,1-classicarKNN(baseClimePCA, 1)))
    print("acerto com k=%s - naiveBayes:%s" %(k,1-NaiveBayes.classificar(m1, m2, v1, v2, baseClimePCA, baseClimePCA.classes)))
    
    
    
