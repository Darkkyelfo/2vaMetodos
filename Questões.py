'''
Created on Aug 13, 2017

@author: raul
'''
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from SepararBase import SepararBase
from Base import Base
from numba import jit
from knnComHamming import KnnComHamming
from NaiveBayes import NaiveBayes
from sklearn.neighbors.kde import KernelDensity
import copy
#Classificador knn para atributos numéricos
def classicarKNN(base,n=1):
    knn = KNeighborsClassifier(n_neighbors=n)
    erroKnn = 0
    loo = LeaveOneOut()
    X = np.array(base.atributos)
    y = np.array(base.classes)
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        knn.fit(X_train,y_train)
        knnPredict = knn.predict(X_test)
        erroKnn = (1-accuracy_score(y_test,knnPredict)) + erroKnn
    return erroKnn/len(baseCancer.classes)

#Função responsável por discretizar os valores númericos
#de uma base
@jit
def discretizacao(base,intervalo=2):
    colunas = np.transpose(base.atributos)
    dic = {}
    inter = []
    c = copy.deepcopy(base.classes)
    a = copy.deepcopy(base.atributos)
    baseDiscretizada = Base(c,a)
    for i,atr in enumerate(colunas):
        dic[i] = [max(atr),min(atr)]
    for e in dic:
        inter.append(float((dic[e][0]-dic[e][1])/(intervalo-1)))
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
            
arq = open("cancer","r")

arq3 = open("wbdc","r")

arq5 = open("wpdc","r")
#Base cancer
dados = arq.readlines()
c = [10]
ig = [0]
classes,atributos = SepararBase.coletarDadosNumericos(dados, c, ig)
baseCancer = Base(classes,atributos)
#pickle.dump(baseCancer,arq2)
#Base wbdc
c = [1]
ig = [0]
dados = arq3.readlines()
classes,atributos = SepararBase.coletarDadosNumericos(dados, c, ig)
baseWbcd = Base(classes,atributos)
#pickle.dump(baseCancer,arq4)
#Base wpdc
dados = arq5.readlines()
classes,atributos = SepararBase.coletarDadosNumericos(dados, c, ig)
baseWpdc = Base(classes,atributos)
#pickle.dump(baseCancer,arq6)
#Fechando arquivos
arq.close()
arq3.close()
arq5.close()

cancerOri = Base(copy.deepcopy(baseCancer.classes),copy.deepcopy(baseCancer.atributos))
wbdcOri = Base(copy.deepcopy(baseWbcd.classes),copy.deepcopy(baseWbcd.atributos))
WpdcOri = Base(copy.deepcopy(baseWpdc.classes),copy.deepcopy(baseWpdc.atributos)) 

#Q1
print("erros 1-nn:")
print("erro base cancer(Hamming):%s"%KnnComHamming.calcular(baseCancer))
print("erro base cancer(Euclidiana):%s"%classicarKNN(baseCancer))
print("erro base wbcd:%s"%classicarKNN(baseWbcd))
print("erro base wpdc:%s"%classicarKNN(baseWpdc))
#Q2
baseCancer2 = Base(copy.deepcopy(cancerOri.classes),copy.deepcopy(cancerOri.atributos))
m1,m2 = separarElementosPorClasse(baseCancer2, ["2","4"])
v1 = np.var(m1)
v2 = np.var(m2)
m1 = np.mean(m1, axis=0)
m2 = np.mean(m2, axis=0)
print("erro naiveBayes:%s"%NaiveBayes.classificar(m1, m2, v1, v2, baseCancer2, ["2","4"]))
#Q3
b = Base(copy.deepcopy(wbdcOri.classes),copy.deepcopy(wbdcOri.atributos))
m1,m2 = separarElementosPorClasse(b, ["M","B"])
v1 = np.var(m1)
v2 = np.var(m2)
m1 = np.mean(m1, axis=0)
m2 = np.mean(m2, axis=0)
print("Sem discretizacao - erro:%s"%(NaiveBayes.classificar(m1, m2, v1, v2, b, ["M","B"])))
intervalos = [ 2, 4,8, 16, 32, 64, 128, 256]
print("erro discretizacao wbdc - NAIVEBAYES")
for e in intervalos:
    b = discretizacao(wbdcOri, e)
    m1,m2 = separarElementosPorClasse(b, ["M","B"])
    v1 = np.var(m1)
    v2 = np.var(m2)
    m1 = np.mean(m1, axis=0)
    m2 = np.mean(m2, axis=0)
    print("intervalo:%s - erro:%s"%(e,NaiveBayes.classificar(m1, m2, v1, v2, b, ["M","B"])))
b = Base(copy.deepcopy(WpdcOri.classes),copy.deepcopy(WpdcOri.atributos))
m1,m2 = separarElementosPorClasse(b, ["N","R"])
v1 = np.var(m1)
v2 = np.var(m2)
m1 = np.mean(m1, axis=0)
m2 = np.mean(m2, axis=0)
print("Sem discretizacao wpdc - erro:%s"%(NaiveBayes.classificar(m1, m2, v1, v2, b,["N","R"])))
print("erro discretizacao wpdc - NAIVEBAYES")
for e in intervalos:
    b = discretizacao(WpdcOri, e)
    m1,m2 = separarElementosPorClasse(b, ["N","R"])
    v1 = np.var(m1)
    v2 = np.var(m2)
    m1 = np.mean(m1, axis=0)
    m2 = np.mean(m2, axis=0)
    print("intervalo:%s - erro:%s"%(e,NaiveBayes.classificar(m1, m2, v1, v2, b,["N","R"])))
#Q5
h = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
X = wbdcOri.atributos
print("janela de pazen gaussian - wbdc")
for i in h:
    kde = KernelDensity(kernel='gaussian', bandwidth=i).fit(X)
    bWbdcPazen = Base(copy.deepcopy(wbdcOri.classes),kde.score_samples(X))
    m1,m2 = separarElementosPorClasse(bWbdcPazen, ["M","B"])
    v1 = np.var(m1)
    v2 = np.var(m2)
    m1 = np.mean(m1, axis=0)
    m2 = np.mean(m2, axis=0)
    print("h:%s - erro:%s"%(i,NaiveBayes.classificar(m1, m2, v1, v2, bWbdcPazen, ["M","B"])))
print("janela de pazen gaussian - wpdc")
X = WpdcOri.atributos
for e in h:
    kde = KernelDensity(kernel='gaussian', bandwidth=i).fit(X)
    bWbdcPazen = Base(copy.deepcopy(WpdcOri.classes),kde.score_samples(X))
    m1,m2 = separarElementosPorClasse(bWbdcPazen, ["N","R"])
    v1 = np.var(m1)
    v2 = np.var(m2)
    m1 = np.mean(m1, axis=0)
    m2 = np.mean(m2, axis=0)
    print("h:%s - erro:%s"%(e,NaiveBayes.classificar(m1, m2, v1, v2,bWbdcPazen,["N","R"])))
#Q4
print("janela de pazen retângular - wbdc")
X = WpdcOri.atributos
for e in h:
    kde = KernelDensity(kernel='tophat', bandwidth=i).fit(X)
    bWbdcPazen = Base(copy.deepcopy(WpdcOri.classes),kde.score_samples(X))
    m1,m2 = separarElementosPorClasse(bWbdcPazen, ["N","R"])
    v1 = np.var(m1)
    v2 = np.var(m2)
    m1 = np.mean(m1, axis=0)
    m2 = np.mean(m2, axis=0)
    print("h:%s - erro:%s"%(e,NaiveBayes.classificar(m1, m2, v1, v2,bWbdcPazen,["N","R"])))