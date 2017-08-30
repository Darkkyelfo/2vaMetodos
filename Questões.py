'''
Created on Aug 13, 2017

@author: raul
'''
from funcoesAuxiliares import *
import numpy as np
from SepararBase import SepararBase
from Base import Base
from knnComHamming import KnnComHamming
from NaiveBayes import NaiveBayes
from sklearn.neighbors.kde import KernelDensity
from JanelaDeParzen import JanelaDeParzen
from PCA import pca, pcaScore
import copy

def classificarNaiveDiscreto(base,classes,intervalos):
    for e in intervalos:
        try:
            b = discretizacao(base, e)
            m1,m2 = separarElementosPorClasse(b, classes)
            v1 = np.cov(np.array(m1).T)
            v2 = np.cov(np.array(m2).T)
            m1 = np.mean(m1, axis=0)
            m2 = np.mean(m2, axis=0)
            print("intervalo:%s - erro:%s"%(e,NaiveBayes.classificar(m1, m2, v1, v2, b, classes)))
        except:
            print("não possivel para o intervalo:%s"%e)
            
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
v1 = np.cov(np.array(m1).T)
v2 = np.cov(np.array(m2).T)
#v1 = np.var(m1)
#v2 = np.var(m2)

m1 = np.mean(m1, axis=0)
m2 = np.mean(m2, axis=0)
print("erro naiveBayes cancer:%s"%NaiveBayes.classificar(m1, m2, v1, v2, baseCancer2, ["2","4"]))
#Q3
b = Base(copy.deepcopy(wbdcOri.classes),copy.deepcopy(wbdcOri.atributos))
m1,m2 = separarElementosPorClasse(b, ["M","B"])
v1 = np.cov(np.array(m1).T)
v2 = np.cov(np.array(m2).T)
m1 = np.mean(m1, axis=0)
m2 = np.mean(m2, axis=0)
print("Sem discretizacao wbdc- erro:%s"%(NaiveBayes.classificar(m1, m2, v1, v2, b, ["M","B"])))
intervalos = [ 2, 4,8, 16, 32, 64, 128, 256]
print("erro discretizacao wbdc - NAIVEBAYES")
classificarNaiveDiscreto(wbdcOri, ["M","B"], intervalos)
#Wpdc sem discretizaçao      
b = Base(copy.deepcopy(WpdcOri.classes),copy.deepcopy(WpdcOri.atributos))
m1,m2 = separarElementosPorClasse(b, ["N","R"])
v1 = np.cov(np.array(m1).T)
v2 = np.cov(np.array(m2).T)
m1 = np.mean(m1, axis=0)
m2 = np.mean(m2, axis=0)
print("Sem discretizacao wpdc - erro:%s"%(NaiveBayes.classificar(m1, m2, v1, v2, b,["N","R"])))

print("erro discretizacao wpdc - NAIVEBAYES")
classificarNaiveDiscreto(WpdcOri,["N","R"], intervalos)
#Q5
'''
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
    print("h:%s - erro:%s"%(i,NaiveBayes.classificar(m1, m2, v1, v2, bWbdcPazen, ["M","B"],"u")))
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
    print("h:%s - erro:%s"%(e,NaiveBayes.classificar(m1, m2, v1, v2,bWbdcPazen,["N","R"],"u")))
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
    print("h:%s - erro:%s"%(e,NaiveBayes.classificar(m1, m2, v1, v2,bWbdcPazen,["N","R"],"u")))
b = pcaScore(cancerOri,0,["2","4"])
b1 = pca(WpdcOri,0)
b2 = pca(wbdcOri)
print("erro base cancer(EuclidianaPCA):%s"%classicarKNN(b))
'''

h = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
m1,m2 = separarElementosPorClasse2(wbdcOri, ["M","B"])
mp1,mp2 = separarElementosPorClasse2(WpdcOri, ["N","R"])
print("naive com janela de pazen gausinana - wbdc")
for i in h:
    print("h:%s erro:%s"%(i,NaiveBayes.classificarParzen(wbdcOri, m1, m2, i, ["M","B"])))
print("naive com janela de pazen gausinana - wpdc")
for i in h:
    print("h:%s erro:%s"%(i,NaiveBayes.classificarParzen(WpdcOri, mp1, mp2, i, ["N","R"])))

print("naive com janela de parzen retangular - wbdc")
for i in h:
    print("h:%s erro:%s"%(i,NaiveBayes.classificarParzen(wbdcOri, m1, m2, i, ["M","B"],"r")))
print("naive com janela de parzen retangular - wpdc")
for i in h:
    print("h:%s erro:%s"%(i,NaiveBayes.classificarParzen(WpdcOri, mp1, mp2, i, ["N","R"])))










