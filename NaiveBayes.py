'''
Created on Aug 18, 2017

@author: raul
'''
import math
import numpy as np
from numba import jit
class NaiveBayes(object):
    '''
    classdocs
    '''

    @staticmethod
    def __px(m,v,x):#m-> media, v-> variancia e x -> vetor
        return (1/(math.sqrt(2*v*math.pi)))*math.exp(-1*(1/(2*v))*np.dot((x-m).T,(x-m)))
    
    @staticmethod
    @jit
    def classificar(m1,m2,v1,v2,base,classes):
        erro = 0
        for i,x in enumerate(base.atributos):
            if(NaiveBayes.__px(m1,v1,x)>NaiveBayes.__px(m2,v2,x)):
                if(base.classes[i] != classes[0]):
                    base.classes[i] = classes[0]
                    erro = erro + 1
            else:
                if(base.classes[i] != classes[1]):
                    base.classes[i] = classes[1]
                    erro = erro + 1
        return float(erro/len(base.classes))
        