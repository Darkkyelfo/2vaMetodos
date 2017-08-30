'''
Created on Aug 29, 2017

@author: raul
'''
import math
import numpy as np
from numba import jit
class JanelaDeParzen(object):
    '''
    classdocs
    '''

    @staticmethod
    @jit
    #base,classe a ser testada, x elemento, h largura da janela e classes a serem escolhidas 
    def gausiano(base,x,h):
        px = 0
        for xi in base.atributos:
            if(x != xi):
                px = JanelaDeParzen.__pGausiano(x,len(base.atributos), xi, h) + px
        return (1/(len(base.atributos))) * px
    
    
    @staticmethod
    @jit
    def retangular(base,x,h):
        px = 0
        for xi in base.atributos:
            if(x != xi):
                px = JanelaDeParzen.__pRetangular(x,len(base.atributos) ,xi, h) + px
        return (1/(len(base.atributos)))*((1/h**len(base.atributos[0]))) * px
    
    @staticmethod
    def __pGausiano(x,n,xi,h):
        #hn = h/math.sqrt(n)
        hn = h
        pid = math.pow((2*math.pi),len(x)) 
        part1 = 1/(math.sqrt(pid*hn**2))
        mediaMenosX = np.subtract(x,xi)
        part2 = -1*(np.linalg.norm(mediaMenosX)**2)/(2*hn**2)
        return part1*math.exp(part2)
    
    
    @staticmethod
    def __pRetangular(x,n,xi,h):
        #h = h/math.sqrt(n)
        mediaMenosX = np.subtract(x,xi)
        fi = np.linalg.norm(mediaMenosX)/h
        if math.fabs(fi) <= 1/2:
            return 1
        return 0
        
        