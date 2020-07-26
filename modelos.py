#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import log, pi, sqrt
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ExpSineSquared
import warnings
import random
warnings.filterwarnings("ignore")


# Função que separa o dataset em treino e teste
def train_testSplit(ids, dataPCA, dataModSULQ, rate_test):
    quantidadeIds = len(ids)
    quantidadeTeste = int(rate_test * quantidadeIds)
    idsTreino, idsTeste = [], []
    while(len(idsTeste)!=quantidadeTeste):
        idEscolhido = random.choice(ids)
        if(idEscolhido not in idsTeste):
            idsTeste.append(idEscolhido)
    for i in range(0, quantidadeIds):
        if(ids[i] not in idsTeste):
            idsTreino.append(ids[i])
    trainPCA, testPCA, trainModSULQ, testModSULQ = [], [], [], []
    for idTreino in idsTreino:
        dataIdPCA = dataPCA.query('id == '+str(idTreino))
        dataIdModSULQ = dataModSULQ.query('id == '+str(idTreino))
        for reg in dataIdPCA.index:
            trainPCA.append(dataIdPCA.loc[reg])
        for reg in dataIdModSULQ.index:
            trainModSULQ.append(dataIdModSULQ.loc[reg])
    for idTeste in idsTeste:
        dataIdPCA = dataPCA.query('id == '+str(idTeste))
        dataIdModSULQ = dataModSULQ.query('id == '+str(idTeste))
        for reg in dataIdPCA.index:
            testPCA.append(dataIdPCA.loc[reg])
        for reg in dataIdModSULQ.index:
            testModSULQ.append(dataIdModSULQ.loc[reg])
    trainPCA = np.array(trainPCA)
    testPCA = np.array(testPCA)
    trainModSULQ = np.array(trainModSULQ)
    testModSULQ = np.array(testModSULQ)
    trainTestPCA = trainPCA[:,1:-1], testPCA[:,1:-1], trainPCA[:,-1], testPCA[:,-1]
    trainTestModSULQ = trainModSULQ[:,1:-1], testModSULQ[:,1:-1], trainModSULQ[:,-1], testModSULQ[:,-1]
    return trainTestPCA, trainTestModSULQ

# Função que computa a função logística (sigmóide)
def sigmoide(row, w):
    yPred = 1/(1+np.exp(-row @ w))
    return yPred

# Função responsável para treinar o modelo de Regressão Logística via Gradiente Descendente
def fitRL(x, y):
    print("[Regressão Logística] Selecionando hiperparâmetros...")

    epochs = [100, 1000, 10000]
    alphas = [0.001, 0.01, 0.1, 1]
    fatoresReg = [0.01, 0.1, 1]
    
    valoresF1 = np.zeros((len(epochs), len(alphas), len(fatoresReg)))

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.25)
    
    auxTrain = np.ones((x_train.shape[0], 1))
    auxVal = np.ones((x_val.shape[0], 1))
    x_train = np.hstack((auxTrain, x_train))
    x_val = np.hstack((auxVal, x_val))
    
    for e in range(0, len(epochs)):
        for a in range(0, len(alphas)):
            for f in range(0, len(fatoresReg)):
                wPrev = np.zeros(x_train.shape[1])
                for ep in range (0, epochs[e]):
                    suma = 0
                    for i in range(0, x_train.shape[0]):
                        suma = suma +  (y_train[i]-sigmoide(x_train[i],wPrev))*np.transpose(x_train[i])
                    wPrevReg = fatoresReg[f]*np.transpose(wPrev)
                    wPrevReg[0] = wPrevReg[0] / fatoresReg[f]
                    w = np.transpose(wPrev) + alphas[a]*((1/x_train.shape[0])*suma - wPrevReg)
                    wPrev = w
                yPredito = []
                for row in x_val:
                    if(sigmoide(row, w)>=0.5):
                        yPredito.append(1)
                    else:
                        yPredito.append(0)
                valoresF1[e,a,f] = f1_score(y_val, yPredito, average='macro')
    
    best_params = np.unravel_index(np.argmax(valoresF1, axis=None), valoresF1.shape)
    n_epochs, alpha, fatorReg = epochs[best_params[0]], alphas[best_params[1]], fatoresReg[best_params[2]]
    
    print("[Regressão Logística] Treinando modelo...")
    
    erroQM = []
    wPrev = np.zeros(x.shape[1]+1)
    aux = np.ones((x.shape[0], 1))
    x = np.hstack((aux, x))
    
    for epochs in range (0, n_epochs):
        suma, sumErro = 0, 0
        for i in range(0, x.shape[0]):
            sumErro = sumErro + (y[i]-sigmoide(x[i],wPrev))**2
            suma = suma +  (y[i]-sigmoide(x[i],wPrev))*np.transpose(x[i])
        wPrevReg = fatorReg*np.transpose(wPrev)
        wPrevReg[0] = wPrevReg[0] / fatorReg
        w = np.transpose(wPrev) + alpha*((1/x.shape[0])*suma - wPrevReg)
        erroEp = ((1/(2*x.shape[0]))*sumErro)
        erroQM.append(erroEp)
        wPrev = w
        
    print("[Regressão Logística] Hiperparâmetros escolhidos para Regressão Logística: {'n_epochs': "+str(n_epochs)+", 'alpha': "+str(alpha)+", 'lambda': "+str(fatorReg)+"}")

    return wPrev, erroQM

# Função responsável para predizer os dados
def predictRL(w, x):
    print("[Regressão Logística] Testando modelo...")
    aux = np.ones((x.shape[0], 1))
    x = np.hstack((aux, x))
    yPredito = [1 if(sigmoide(row, w)>=0.5) else 0 for row in x]
    return yPredito

# Função responsável para "treinar" que gera os dados estatísticos necessários para o modelo de 
# Análise de Discriminante Gaussiano
def fitAGD(x, y):
    print("[Análise de Discriminante Gaussiano] Treinando modelo...")
    classes, occurrences = np.unique(y, return_counts=True)
    numClasses = len(classes)
    n = len(y)
    numFeatures = x.shape[1]
    
    probabilidadeClasses = dict(zip(classes, occurrences))

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
    probabilities = np.zeros(model['numClasses'])
    for classe in model['classes']:
        classe = int(classe)
        fator1 = 1/((sqrt(np.linalg.det(model['covar'][:, :, classe])) * ((2*pi)**(model['numFeatures']/2)))+10**-6) 
        
        inversa = np.linalg.inv(model['covar'][:, :, classe])
        difXMedia = row - model['media'][:, classe]
        z = (-0.5) * (np.transpose(difXMedia) @ inversa @ difXMedia)
        probabilities[classe] = fator1 * np.exp(z)
    return model['classes'][np.argmax(probabilities)]
    
# Função utilizada para predizer as classes de um conjunto de registros
def predictAGD(model, x_test):
    print("[Análise de Discriminante Gaussiano] Testando modelo...")
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
    results = [[function(x[i], x_teste), y[i]] for i in range(0, x.shape[0])]
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
def fitAD(x, y):
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
    configSVM = [{'kernel': ['rbf'], 'C': 2 ** np.array([-1.0, -2.0, -3.0, 0, 1, 2]), 'gamma': 2 ** np.array([-3.0, -2.0, -1.0])},
                 {'kernel':['poly'], 'C': 2 ** np.array([-1.0, -2.0, -3.0, 0, 1, 2]),'degree': np.array([3,4,5])}]

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
    parameters = {'kernel':(RBF(1.0),Matern(length_scale=1.0, nu=1.5), RationalQuadratic(length_scale=1.0, alpha=1.5))}

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