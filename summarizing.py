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
from mmr import tfidf
from mmr import calcidf
from mmr import calctfidf
from mmr import cosineSimilarity
from mmr import mmrReorder
import math


def generate_summary_bylength(sentences,word,node,filen,outpath,lengths):
    #sumLengths = [20,30,50]
    sumLengths = lengths
    outDirectory = outpath+filen[:-4]
    # modelSum = []
    for i in sumLengths:
        '''
        i=sumLengths[0]
        '''
        summarytext = []
        sumLength = i
        #fileoutputname = filename[:-8] + "out" + str(i) + ".txt"
        if not os.path.exists(outDirectory):
            os.makedirs(outDirectory)
        fileoutputname = str(i) + ".txt"
        outFile = open(os.path.join(outDirectory, fileoutputname), "w", encoding="utf-8")
        #summarytext = ' '.join(sentence[:i,0])
        currentLenght = 0
        for idx,x in enumerate(sentences):
            '''
            idx=0
            x=sentences[idx]
            '''
            # memilih text dari scora tertinggi
            senLength = len(nltk.word_tokenize(x[1]))
            if(senLength < (sumLength-currentLenght)):
                #summarytext += x[0] + '. '
                summarytext.append(x)
                currentLenght += senLength
            # reorder rank mmr
            
        summarytext= sorted(summarytext,key=lambda x: x[0])
        text=''
        for x in summarytext:
            text += x[1] + '.\n '
        outFile.write(text)
        outFile.close()
        # modelSum.append(summarytext)

def generate_summary_bycompression(sentence,filen,outpath,compression,textlength):
    #sumLengths = [20,30,50]
    sumLengths = 100 - np.array(compression)
    outDirectory = outpath+filen[:-4]
    # modelSum = []
    for i in sumLengths:
        summarytext = []
        sumLength = round(textlength*i/100)
        #fileoutputname = filename[:-8] + "out" + str(i) + ".txt"
        if not os.path.exists(outDirectory):
            os.makedirs(outDirectory)
        fileoutputname = str(100-i) + "%.txt"
        outFile = open(os.path.join(outDirectory, fileoutputname), "w", encoding="utf-8")
        #summarytext = ' '.join(sentence[:i,0])
        currentLenght = 0
        for x in sentence:
            senLength = len(nltk.word_tokenize(x[1]))
            if(senLength < (sumLength-currentLenght)):
                #summarytext += x[0] + '. '
                summarytext.append(x)
                currentLenght += senLength
        summarytext= sorted(summarytext,key=lambda x: x[0])
        text=''
        for x in summarytext:
            text += x[1] + '.\n '
        outFile.write(text)
        outFile.close()
        # modelSum.append(summarytext)
        
def generate_summary_bylength_mmr(sentences,word,node,filen,outpath,lengths,n):
    #sumLengths = [20,30,50]
    #tf_idf=tfidf(sentences)
    idf =calcidf(sentences)
    sumLengths = lengths
    outDirectory = outpath+filen[:-4]
    # modelSum = []
    for i in sumLengths:
        '''
        i=sumLengths[3]
        '''
        summarytext = []
        sumLength = i
        #fileoutputname = filename[:-8] + "out" + str(i) + ".txt"
        if not os.path.exists(outDirectory):
            os.makedirs(outDirectory)
        fileoutputname = str(i) + ".txt"
        outFile = open(os.path.join(outDirectory, fileoutputname), "w", encoding="utf-8")
        #summarytext = ' '.join(sentence[:i,0])
        sentencesmmr = [[x[0],x[1],float(x[2]),0,x[2]] for x in sentences] ### 3: used sentence, 4: textrank score, 2: textrank -> mmrscore
        currentLenght = 0
        for idx in range(0,len(sentencesmmr)):
            '''
            idx=0
            '''
            x=sentencesmmr[0]
            # reorder rank mmr
            if(idx!=0):
                sentencesmmr = mmrReorder(sentencesmmr,n)
                sentencesmmr = sorted(sentencesmmr,key=lambda x: (x[3], -x[2]))
                x=sentencesmmr[0]
                
            # memilih text dari scora tertinggi
            senLength = len(nltk.word_tokenize(x[1]))
            if(senLength < (sumLength-currentLenght)):
                #summarytext += x[0] + '. '
                sentencesmmr[0][3]=1
                summarytext.append(x)
                currentLenght += senLength
            else:
                #print(idx)
                sentencesmmr[0][3]=2
            
            
        summarytext= sorted(summarytext,key=lambda x: x[0])
        text=''
        for x in summarytext:
            text += x[1] + ' '
        outFile.write(text)
        outFile.close()
        # modelSum.append(summarytext)

def generate_summary_bycompression_mmr(sentences,word,node,filen,outpath,compression,textlength,n):
    #sumLengths = [20,30,50]
    #tf_idf=tfidf(sentences)
    idf =calcidf(sentences)
    sumLengths = 100 - np.array(compression)
    outDirectory = outpath+filen[:-4]
    # modelSum = []
    for i in sumLengths:
        '''
        i=sumLengths[3]
        '''
        summarytext = []
        sumLength = round(textlength*i/100)
        #fileoutputname = filename[:-8] + "out" + str(i) + ".txt"
        if not os.path.exists(outDirectory):
            os.makedirs(outDirectory)
        fileoutputname = str(100-i) + "%.txt"
        outFile = open(os.path.join(outDirectory, fileoutputname), "w", encoding="utf-8")
        #summarytext = ' '.join(sentence[:i,0])
        sentencesmmr = [[x[0],x[1],float(x[2]),0,x[2]] for x in sentences] ### 3: used sentence, 4: textrank score, 2: textrank -> mmrscore
        currentLenght = 0
        for idx in range(0,len(sentencesmmr)):
            '''
            idx=0
            '''
            x=sentencesmmr[0]
            # reorder rank mmr
            if(idx!=0):
                sentencesmmr = mmrReorder(sentencesmmr,n,idf)
                sentencesmmr = sorted(sentencesmmr,key=lambda x: (x[3], -x[2]))
                x=sentencesmmr[0]

            # memilih text dari scora tertinggi
            senLength = len(nltk.word_tokenize(x[1]))
            if(senLength < (sumLength-currentLenght)):
                #summarytext += x[0] + '. '
                sentencesmmr[0][3]=1
                summarytext.append(x)
                currentLenght += senLength
            else:
                #print(idx)
                sentencesmmr[0][3]=2


        summarytext= sorted(summarytext,key=lambda x: x[0])
        text=''
        for x in summarytext:
            text += x[1] + ' '
        outFile.write(text)
        outFile.close()
        # modelSum.append(summarytext)


def sentence_weight(sentence,word,nodes):
    ##### sentence weighting
    text_n = len(word)
    sentenceWeight = np.zeros(text_n)
    for i,x in enumerate(word):
        l=len(sentence[i].split(' '))
        #print(l)
        sums = 0
        j=0
        for y in x:
            sums += nodes[nodes[:,0]==y,2][0]
            j+=1
        if j>0:
            #sentenceWeight[i]=sums/j ### average based
            #sentenceWeight[i]=sums ### score based
            #sentenceWeight[i]=sums/(1+math.log10(j)) ### log based
            sentenceWeight[i]=sums/l ### sentence average based
            #print(sums)
    
    sentence = [[i, sentence[i],sentenceWeight[i]] for i,x in enumerate(word)]
    #sentence = np.sort(np.array(sentence,dtype=object),axis=-0)
    sentence = sorted(sentence,key=lambda x: -x[2])
    sentence = np.array(sentence)
    return sentence
    
        
if __name__ == "__main__":
    path = "data/body/"
    filens = os.listdir(path)
    stops = stopwords.words('english')
    sentence, word = preprocess(path+filens[0])
    nodes = textrank(word)
    generate_summary_bylength(sentence,word,nodes,filens[0])
