'''
Created on Aug 18, 2017

@author: raul
'''
import math
import numpy as np
from numba import jit
from JanelaDeParzen import JanelaDeParzen
class NaiveBayes(object):
    '''
    classdocs
    '''
    '''
    @staticmethod
    def __px(m,v,x):#m-> media, v-> variancia e x -> vetor
        return (1/(math.sqrt(2*v*math.pi)))*math.exp(-1*(1/(2*v))*np.dot((x-m).T,(x-m)))
    '''
    
    @staticmethod
    def __px(m,v,x):#m-> media, v-> covariancia e x -> vetor
        pid = math.pow((2*math.pi),len(x)) 
        det = np.linalg.det(v)
        part1 = 1/(math.sqrt(pid*det))
        mediaMenosX = np.subtract(x,m)
        part2 = (-1/2)*np.dot(np.dot(mediaMenosX.T,np.linalg.inv(v)),mediaMenosX)
        return part1 * math.exp(part2)

    @staticmethod
    def __pxu(m,v,x):
        return  (1/(v*math.sqrt(2*math.pi)))*math.exp(-1*(1/2)*math.pow(((x-m)/v),2))
    
    @staticmethod
    @jit
    def classificar(m1,m2,v1,v2,base,classes,modo="m"):
        erro = 0
        for i,x in enumerate(base.atributos):
            if(modo == "m"):
                if(NaiveBayes.__px(m1,v1,x)>NaiveBayes.__px(m2,v2,x)):
                    if(base.classes[i] != classes[0]):
                        base.classes[i] = classes[0]
                        erro = erro + 1
                else:
                    if(base.classes[i] != classes[1]):
                        base.classes[i] = classes[1]
                        erro = erro + 1
            if(modo=="u"):
                if(NaiveBayes.__pxu(m1,v1,x)>NaiveBayes.__pxu(m2,v2,x)):
                    if(base.classes[i] != classes[0]):
                        base.classes[i] = classes[0]
                        erro = erro + 1
                else:
                    if(base.classes[i] != classes[1]):
                        base.classes[i] = classes[1]
                        erro = erro + 1
        return float(erro/len(base.classes))
    
    @staticmethod
    @jit
    def classificarParzen(base,baseC1,baseC2,h,classes,modo="g"):
        erro = 0
        for i,x in enumerate(base.atributos):
            if(modo=="g"): #janela gausinana
                if(JanelaDeParzen.gausiano(baseC1, x, h) > JanelaDeParzen.gausiano(baseC2, x, h)):
                    if(base.classes[i] != classes[0]):
                        base.classes[i] = classes[0]
                        erro = erro + 1
                else:
                    if(base.classes[i] != classes[1]):
                        base.classes[i] = classes[1]
                        erro = erro + 1
            if(modo == "r"):
                if(JanelaDeParzen.retangular(baseC1, x, h) > JanelaDeParzen.retangular(baseC2, x, h)):
                    if(base.classes[i] != classes[0]):
                        base.classes[i] = classes[0]
                        erro = erro + 1
                else:
                    if(base.classes[i] != classes[1]):
                        base.classes[i] = classes[1]
                        erro = erro + 1
        return erro/(len(base.classes))
        