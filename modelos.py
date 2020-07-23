#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, pi, sqrt
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared
import warnings
import random
warnings.filterwarnings("ignore")


# Função que separa o dataset em treino e teste
def train_testSplit(ids, data, rate_test):
    quantidadeIds = len(ids)
    quantidadeTeste = int(rate_test * quantidadeIds)
    idsTreino = []
    idsTeste = []
    while(len(idsTeste)!=quantidadeTeste):
        idEscolhido = random.choice(ids)
        if(idEscolhido not in idsTeste):
            idsTeste.append(idEscolhido)
    for i in range(0, quantidadeIds):
        if(ids[i] not in idsTeste):
            idsTreino.append(ids[i])
    train = []
    test = []
    for idTreino in idsTreino:
        dataId = data.query('id == '+str(idTreino))
        for reg in dataId.index:
            train.append(dataId.loc[reg])
    for idTeste in idsTeste:
        dataId = data.query('id == '+str(idTeste))
        for reg in dataId.index:
            test.append(dataId.loc[reg])
    train = np.array(train)
    test = np.array(test)
    return train[:,1:-1], test[:,1:-1], train[:,-1], test[:,-1]

# Função que computa a função logística (sigmóide)
def sigmoide(row, w):
    yPred = 1/(1+np.exp(-row @ w))
    return yPred

# Função responsável para treinar o modelo de Regressão Logística via Gradiente Descendente
def fitRL(x, y, n_epochs, alpha, fatorReg):
    print("[Regressão Logística] Treinando modelo...")

    erroQM = []
    wPrev = np.zeros(x.shape[1]+1)
    aux = np.ones((x.shape[0], 1))
    x = np.hstack((aux, x))
    
    for epochs in range (0, n_epochs):
        suma = 0
        sumErro = 0
        for i in range(0, x.shape[0]):
            sumErro = sumErro + (y[i]-sigmoide(x[i],wPrev))**2
            suma = suma +  (y[i]-sigmoide(x[i],wPrev))*np.transpose(x[i])
        wPrevReg = fatorReg*np.transpose(wPrev)
        wPrevReg[0] = wPrevReg[0] / fatorReg
        w = np.transpose(wPrev) + alpha*((1/x.shape[0])*suma - wPrevReg)
        erroEp = ((1/(2*x.shape[0]))*sumErro)
        erroQM.append(erroEp)
        wPrev = w
    return wPrev, erroQM

# Função responsável para predizer os dados
def predictRL(w, x):
    print("[Regressão Logística] Testando modelo...")
    yPredito = []
    aux = np.ones((x.shape[0], 1))
    x = np.hstack((aux, x))
    for row in x:
        if(sigmoide(row, w)>=0.5):
            yPredito.append(1)
        else:
            yPredito.append(0)
    return yPredito

# Função responsável para "treinar" que gera os dados estatísticos necessários para o modelo de 
# Análise de Discriminante Gaussiano
def fitAGD(x, y):
    print("[Análise Discriminante Gaussiano] Treinando modelo...")
    classes, ocorrencs = np.unique(y, return_counts=True)
    numClasses = len(classes)
    n = len(y)
    numFeatures = x.shape[1]
    
    probabilidadeClasses = dict(zip(classes, ocorrencs))
    for key in probabilidadeClasses:
        probabilidadeClasses[key] = probabilidadeClasses[key] / n
    
    media = np.zeros((numFeatures, numClasses))
    covar = np.zeros((numFeatures, numFeatures, numClasses))

    for classe in classes:
        xk = x[y == classe]
        classe = int(classe)
        media[:, int(classe)] = np.mean(xk, axis=0)
        xi_mean = xk - media[:, int(classe)]
        covar[:, :, int(classe)] = (np.transpose(xi_mean) @ xi_mean)/len(xk)
        covar[:, :, int(classe)] += np.eye(numFeatures) * np.mean(np.diag(covar[:,:,classe]))  * 10 ** -6
    return {'media': media, 'covar': covar, 'classes': classes, 'numRows': n, 'numClasses': numClasses, 'numFeatures': numFeatures, 'probabilidadeClasses': probabilidadeClasses }

# Função responsável para predizer a classe de um único registro
def predict1AGD(model, row):
    probabilits = np.zeros(model['numClasses'])
    for classe in model['classes']:
        classe = int(classe)
        fator1 = 1/((sqrt(np.linalg.det(model['covar'][:, :, classe])) * ((2*pi)**(model['numFeatures']/2)))+10**-6) 
        
        inversa = np.linalg.inv(model['covar'][:, :, classe])
        difXMedia = row - model['media'][:, classe]
        z = (-0.5) * (np.transpose(difXMedia) @ inversa @ difXMedia)
        probabilits[classe] = fator1 * np.exp(z)
    return model['classes'][np.argmax(probabilits)]
    
# Função utilizada para predizer as classes de um conjunto de registros
def predictAGD(model, x_test):
    print("[Análise Discriminante Gaussiano] Testando modelo...")
    yPredito = np.array([predict1AGD(model, row) for row in x_test])
    return yPredito

# Funções que realizam os cálculos de distância entre dois registros
def distance_euclidian(x1, x2):
    return sqrt(np.sum([abs(i - j) for i, j in zip(x1,x2)]))

def distance_manhattan(x1, x2):
    return np.sum([abs(i-j) for i, j in zip(x1,x2)])

# Função responsável por predizer a classe de um único registro
def predict1KNN(x, y, x_teste, k, function):
    classes = np.unique(y)
    results = []
    for i in range(0, x.shape[0]):
        results.append([function(x[i], x_teste), y[i]])
    results = sorted(results)
    dictClasses = {}
    for i in classes:
        dictClasses[i] = 0
    for i in range(0, k):
        for row in dictClasses.keys():
            if results[i][1] == row:
                dictClasses[row] += 1

    minimus = [results[i][1] for i in range (0,k)]
    
    contClasses = [(x, minimus.count(x)) for x in set(minimus)]

    maximo = np.argmax(contClasses, axis=0)

    return contClasses[maximo[1]][0]

# Função utilizada para predizer as classes de um conjunto de registros
def predictKNN(x, y, x_test, function):
    print("[KNN] Treinando modelo...")
    lista_k = [3,5,7,9,11]
    hiperparamKNN = {'n_neighbors': lista_k}
    
    model = GridSearchCV(KNeighborsClassifier(), hiperparamKNN)
    
    model.fit(x, y)
    
    params = model.best_params_
    print("[KNN] Hiperparâmetros escolhidos para KNN: ", model.best_params_)
    
    print("[KNN] Testando modelo...")
    yPredito = [predict1KNN(x, y, row, model.best_params_['n_neighbors'], function) for row in x_test]

    return yPredito

# Função responsável por treinar a Árvore de Decisão e escolher os melhores hiperparâmetros por meio de grid-search
def fitAD(x, y, criterion, max_depth):
    print("[Árvore de Decisão] Selecionando hiperparâmetros...")
    listMaxDepth = range(1, 50)
    configTree = {'criterion':['gini','entropy'],'max_depth':listMaxDepth}
    clf = GridSearchCV(DecisionTreeClassifier(), configTree)

    print("[Árvore de Decisão] Treinando modelo...")
    clf.fit(x, y)
    
    params = clf.best_params_
    print("[Árvore de Decisão] Hiperparâmetros escolhidos para Árvore de Decisão: ", clf.best_params_)

    return clf

# Função que prediz a classe de um conjunto ou um único registro
def predictAD(tree, x, y, x_test):
    print("[Árvore de Decisão] Testando modelo...")
    yPredito = tree.predict(x_test)
    return yPredito

# Função responsável por treinar o SVM e escolher os melhores hiperparâmetros por meio de grid-search
def fitSVM(x, y):
    print("[SVM] Selecionando hiperparâmetros...")
    configSVM = [{'kernel': ['rbf'], 'C': 2 ** np.arange(-5.0, 16.0, 4), 'gamma': 2 ** np.arange(-15.0, 4.0, 4)},
                 {'kernel':['poly'], 'C': 2 ** np.arange(-5.0, 16.0, 4),'degree': np.arange(2, 6)},
                 {'kernel': ['linear'], 'C': 2 ** np.arange(-5.0, 16.0, 4)}]

    model = GridSearchCV(SVC(), configSVM)
 
    print("[SVM] Treinando modelo...")
    model.fit(x, y)

    print("[SVM] Hiperparâmetros escolhidos para SVM: ", model.best_params_)

    return model
    

# Função que prediz a classe de um conjunto ou um único registro
def predictSVM(svm, x, y, x_test):
    print("[SVM] Testando modelo...")
    yPredito = svm.predict(x_test)
    return yPredito

# Função responsável por treinar o Random Forest e escolher os melhores hiperparâmetros por meio de grid-search
def fitRF(x, y):
    print("[Random Forest] Selecionando hiperparâmetros...")
    listEstimators = range(100, 150)
    listMaxDepth = range(1, 10)
   
    configRandom = {'criterion':['gini','entropy'],'n_estimators':listEstimators, 'max_depth':listMaxDepth}
    clf = GridSearchCV(RandomForestClassifier(), configRandom)

    print("[Random Forest] Treinando modelo...")
    clf.fit(x,y)

    print("[RANDOM FOREST] Hiperparâmetros escolhidos para Radom Forest: ", clf.best_params_)

    return clf

# Função que prediz a classe de um conjunto ou um único registro
def predictRF(randomForest, x, y, x_test):
    print("[Random Forest] Testando modelo...")
    yPredito = randomForest.predict(x_test)
    return yPredito

# Função responsável por treinar o Processo Gaussiano e escolher o melhor kernel por meio de grid-search
def fitGP(x, y):
    print("[Processo Gaussiano] Selecionando kernel...")
    parameters = {'kernel':(RBF(1.0),Matern(length_scale=1.0, nu=1.5), RationalQuadratic(length_scale=1.0, alpha=1.5), ExpSineSquared(length_scale=1, periodicity=1))}
   
    clf = GridSearchCV(GaussianProcessClassifier(), parameters)

    print("[Processo Gaussiano] Treinando modelo...")
    clf.fit(x,y)

    print("[Processo Gaussiano] Kernel escolhidos para Processo Gaussiano: ", clf.best_params_)

    return clf

# Função que prediz a classe de um conjunto ou um único registro
def predictGP(gpc, x, y, x_test):
    print("[Processo Gaussiano] Testando modelo...")
    yPredito = gpc.predict(x_test)
    return yPredito