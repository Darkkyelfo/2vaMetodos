'''
Created on Aug 16, 2017

@author: raul
'''

from sklearn.metrics import accuracy_score
class KnnComHamming(object):
    '''
    classdocs
    '''
    
    @staticmethod
    def calcular(base,k=1):
        erroKnn = 0
        for i,e in enumerate(base.atributos):
            selecionado = [base.classes[i],e]
            del base.atributos[i]
            del base.classes[i]
            distancias = []
            for i1,e2 in enumerate(base.atributos):
                d = KnnComHamming.distanciaHamming(selecionado[1], e2)
                if(d>0):
                    distancias.append([d,i1,base.classes[i1]])
            distancias.sort(key=lambda x:x[0], reverse=False)
            qtMaisApareceu=0
            classe = ""
            for i in distancias[0:k]:
                    cont = KnnComHamming.__contarClasses(distancias[0:k], i[2])
                    if(cont > qtMaisApareceu):
                        qtMaisApareceu = cont
                        classe = i[2]
            erroKnn = (1-accuracy_score([selecionado[0]],[classe])) + erroKnn
            selecionado[0] = classe
            base.classes.append(selecionado[0])
            base.atributos.append(selecionado[1])
        return float(erroKnn/len(base.classes))
                
    @staticmethod
    def distanciaHamming(atr1,atr2):
        satr1 = KnnComHamming.arrayParaString(atr1)
        satr2 = KnnComHamming.arrayParaString(atr2)
        if(len(satr1)==len(satr2)):
            return sum(ch1 != ch2 for ch1, ch2 in zip(atr1, atr2))
        return -1
    
    @staticmethod
    def arrayParaString(array):
        saida = ""
        for i in array:
            saida = str(i)+saida
        return saida
    
    @staticmethod
    def __contarClasses(distancias,nomeClasse):
        cont = 0
        for i in distancias:
            if(i[2]==nomeClasse):
                cont+=1
        return cont
        