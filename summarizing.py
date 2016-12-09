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
from textrank import textrank

def generate_summary(sentence,word,nodes,filen):
    ##### sentence weighting
    text_n = len(word)
    sentenceWeight = np.zeros(text_n)
    for i,x in enumerate(word):
        sum = 0
        j=0
        for y in x:
            sum += nodes[nodes[:,0]==y,2][0]
            j+=1
        if j>0:
            sentenceWeight[i]=sum/j

    sentence = [[sentence[i],sentenceWeight[i]] for i,x in enumerate(word)]
    #sentence = np.sort(np.array(sentence,dtype=object),axis=-0)
    sentence = sorted(sentence,key=lambda x: x[1])
    sentence = np.array(sentence)
    
    ##### generate summary, summary length
    sumLengths = [20,30,50]
    outDirectory = "data/sys-summary/"+filen[:-4]
    # modelSum = []
    for i in sumLengths:
        summarytext = ''
        sumLength = i
        #fileoutputname = filename[:-8] + "out" + str(i) + ".txt"
        if not os.path.exists(outDirectory):
            os.makedirs(outDirectory)
        fileoutputname = str(i) + ".txt"
        outFile = open(os.path.join(outDirectory, fileoutputname), "w", encoding="utf-8")
        #summarytext = ' '.join(sentence[:i,0])
        currentLenght = 0
        for x in sentence:
            senLength = len(nltk.word_tokenize(x[0]))
            if(senLength < (sumLength-currentLenght)):
                summarytext += x[0] + ' '
                currentLenght += senLength
        outFile.write(summarytext)
        outFile.close()
        # modelSum.append(summarytext)
        
path = "data/body/"
filens = os.listdir(path)
stops = stopwords.words('english')
sentence, word = preprocess(path+filens[0])
nodes = textrank(word)
generate_summary(sentence,word,nodes,filens[0])
