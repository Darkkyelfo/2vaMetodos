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
from PCA import pca
import copy
from sklearn.metrics import confusion_matrix

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
            print(confusion_matrix(base.classes, b.classes,classes))
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
cancerOriSort = Base(copy.deepcopy(baseCancer.classes),copy.deepcopy(baseCancer.atributos))
wbdcOriSort = Base(copy.deepcopy(baseWbcd.classes),copy.deepcopy(baseWbcd.atributos))
WpdcOriSort = Base(copy.deepcopy(baseWpdc.classes),copy.deepcopy(baseWpdc.atributos)) 

#Q1
print("erros 1-nn:")
erro,pred,certa = KnnComHamming.calcular(baseCancer)
print("erro base cancer(Hamming):%s"%erro)
print(confusion_matrix(certa,pred))
erro,pred,certa = classicarKNN(cancerOriSort, 1)
print("erro base cancer(Euclidiana):%s"%erro)
print(confusion_matrix(certa,pred))
erro,pred,certa = classicarKNN(baseWbcd)
print("erro base wbcd:%s"%erro)
print(confusion_matrix(certa,pred,["M","B"]))
erro,pred,certa = classicarKNN(baseWpdc)
print("erro base wpdc:%s"%erro)
print(confusion_matrix(certa,pred,["N","R"]))
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
print(confusion_matrix(cancerOri.classes, baseCancer2.classes,["2","4"]))
#Q3
b = Base(copy.deepcopy(wbdcOri.classes),copy.deepcopy(wbdcOri.atributos))
m1,m2 = separarElementosPorClasse(b, ["M","B"])
v1 = np.cov(np.array(m1).T)
v2 = np.cov(np.array(m2).T)
m1 = np.mean(m1, axis=0)
m2 = np.mean(m2, axis=0)
print("Sem discretizacao wbdc- erro:%s"%(NaiveBayes.classificar(m1, m2, v1, v2, b, ["M","B"])))
print(confusion_matrix(wbdcOri.classes, b.classes,["M","B"]))
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
print(confusion_matrix(WpdcOri.classes, b.classes,["N","R"]))

print("erro discretizacao wpdc - NAIVEBAYES")
classificarNaiveDiscreto(WpdcOri,["N","R"], intervalos)
#Q5

h = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
m1,m2 = separarElementosPorClasse2(wbdcOri, ["M","B"])
mp1,mp2 = separarElementosPorClasse2(WpdcOri, ["N","R"])
print("naive com janela de pazen gausinana - wbdc")
for i in h:
    wbdcCopia = Base(copy.deepcopy(wbdcOri.classes),copy.deepcopy(wbdcOri.atributos))
    print("h:%s erro:%s"%(i,NaiveBayes.classificarParzen(wbdcCopia, m1, m2, i, ["M","B"])))
    print(confusion_matrix(wbdcOri.classes, wbdcCopia.classes,["M","B"]))
print("naive com janela de pazen gausinana - wpdc") 
for i in h:
    wpdcCopia = Base(copy.deepcopy(WpdcOri.classes),copy.deepcopy(WpdcOri.atributos))
    print("h:%s erro:%s"%(i,NaiveBayes.classificarParzen(wpdcCopia, mp1, mp2, i, ["N","R"])))
    print(confusion_matrix(WpdcOri.classes, wpdcCopia.classes,["N","R"]))
    
print("naive com janela de parzen retangular - wbdc")
for i in h:
    wbdcCopia = Base(copy.deepcopy(wbdcOri.classes),copy.deepcopy(wbdcOri.atributos)) 
    print("h:%s erro:%s"%(i,NaiveBayes.classificarParzen(wbdcCopia, m1, m2, i, ["M","B"],"r"))) 
    print(confusion_matrix(wbdcOri.classes, wbdcCopia.classes,["M","B"]))
print("naive com janela de parzen retangular - wpdc")
for i in h:
    wpdcCopia = Base(copy.deepcopy(WpdcOri.classes),copy.deepcopy(WpdcOri.atributos))
    print("h:%s erro:%s"%(i,NaiveBayes.classificarParzen(wpdcCopia, mp1, mp2, i, ["N","R"])))
    print(confusion_matrix(WpdcOri.classes, wpdcCopia.classes,["N","R"]))

#Q6
wbdcPCA = pca(wbdcOri, len(wbdcOri.atributos[0])-1)
m1,m2 = separarElementosPorClasse(wbdcPCA, ["M","B"])
v1 = np.var(m1)
v2 = np.var(m2)
m1 = np.mean(m1, axis=0)
m2 = np.mean(m2, axis=0)
print("erro wbdc naiveBayes univariado:%s"%NaiveBayes.classificar(m1, m2, v1, v2, wbdcPCA,["M","B"],"u"))
print(confusion_matrix(wbdcOriSort.classes, wbdcPCA.classes))

WpdcPCA = pca(WpdcOri,len(WpdcOri.atributos[0])-1)
m1,m2 = separarElementosPorClasse(WpdcPCA, ["N","R"])
v1 = np.var(m1)
v2 = np.var(m2)
m1 = np.mean(m1, axis=0)
m2 = np.mean(m2, axis=0)
print("erro wpdc naiveBayes univariado:%s"%NaiveBayes.classificar(m1, m2, v1, v2, WpdcPCA,["N","R"],"u"))
print(confusion_matrix(WpdcOriSort.classes, WpdcPCA.classes))











