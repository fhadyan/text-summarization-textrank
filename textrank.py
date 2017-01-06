import os
from platform import node
import copy
import nltk
import re
import numpy as np
from pyrouge import Rouge155
from pprint import pprint
from numpy import genfromtxt
import csv
import shutil
from nltk.corpus import stopwords
from preprocessing import preprocess

def textrank(word):
    ##### create graph
    nodes = []
    temp = []
    [temp.extend(x) for x in word]
    [nodes.append(x) for x in temp if x not in nodes]

    edges = []
    for x in word:
        edges.extend([[x[idx],x[idx+1]] for idx,y in enumerate(x) if idx<len(x)-1])

    edgesWeight =[]
    #edgesWeight = [[x[0],x[1], edges.count(x)] for x in edges if x not in edgesWeight]
    for x in edges:
        if x not in edgesWeight:
            edgesWeight.append(x)
    edgesWeight = [[x[0], x[1] ,edges.count(x)] for x in edgesWeight]
    edgesWeight = np.array(edgesWeight, dtype=object)
    # temp = [edges.count(x) for x in edges]

    ##### create adjency matrix
    # initValue = 1 ### textrank 1
    initValue = 1/len(word)
    nodes = [[x, i, initValue] for i,x in enumerate(nodes)]
    nodes = np.array(nodes, dtype=object)

    nodes_n = len(nodes)
    adjMatrix = np.zeros((nodes_n,nodes_n))
    for i in range(0,len(nodes)-1):
        x = nodes[i,:]
        out = edgesWeight[edgesWeight[:,0]==x[0],:]
        for y in out:
            out_i = nodes[nodes[:,0]==y[1],:]
            adjMatrix[out_i[0][1],i] = y[2]

    ##### dangling nodes
    colSum = np.sum(adjMatrix,axis=0)
    #adjMatrix[np.where(colSum==0),:]=1
    adjMatrix[:,np.where(colSum==0)]=1

    ##### Textrank
    d = 0.85
    #d = 0.95
    e = 1
    t  = 0.0001
    while e>t:
        wVn = copy.deepcopy(nodes[:,2])
        for i,x in enumerate(nodes):
            inVi = adjMatrix[x[1],:]
            if np.sum(inVi)>0:
                wjk = []
                for y in np.where(inVi>0)[0]:#
                    outVj = adjMatrix[:,y]
                    wjk.append([np.sum(outVj)])
                wVj = nodes[np.where(inVi>0),2][0]
                inVi = inVi[inVi[:]>0]
                nodes[i,2] = (1-d) + (d*np.sum([z/wjk[j]*wVj[j] for j,z in enumerate(inVi)]))
            else:
                nodes[i,2] = (1-d)
        wVn1 = nodes[:,2]
        er = [abs(float(wVn[j])-float(wVn1[j])) for j,z in enumerate(wVn)]
        e = max(er)
        
    return nodes

if __name__ == "__main__":       
    path = "data/body/"
    filens = os.listdir(path)
    stops = stopwords.words('english')
    sentence, word = preprocess(path+filens[0])
    textrank(word)