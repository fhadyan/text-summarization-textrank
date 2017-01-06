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
from summarizing import generate_summary_bylength
from summarizing import generate_summary_bycompression
from summarizing import sentence_weight

if __name__ == "__main__": 
    # dataset 2
    l = [50,100,150,200] 
    outDir = 'data/sys-summary-baseline-length/'
    if os.path.exists(outDir):
        shutil.rmtree(outDir)
        os.makedirs(outDir)
    else:
        os.makedirs(outDir)
    
    path = "data/body/"
    filens = os.listdir(path)
    stops = stopwords.words('english')
    for idf,filen in enumerate(filens[0:100]):
        #filen=filens[0]
        print(idf)
        fpath=path+filen
        sentence, word, textlength = preprocess(fpath,stops)
        outDirectory = outDir+filen[:-4]
        for n in l:
            summarytext = ''
            sumLength = n
            if not os.path.exists(outDirectory):
                os.makedirs(outDirectory)
            fileoutputname = str(n) + ".txt"
            outFile = open(os.path.join(outDirectory, fileoutputname), "w", encoding="utf-8")
            currentLenght = 0
            for x in sentence:
                senLength = len(nltk.word_tokenize(x))
                if(senLength < (sumLength-currentLenght)):
                    summarytext += x + ' '
                    currentLenght += senLength
            outFile.write(summarytext)
            outFile.close()
            # modelSum.append(summarytext)
            
        