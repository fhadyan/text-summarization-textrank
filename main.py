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
from summarizing import generate_summary_bylength_mmr
from summarizing2 import generate_summary_bycompression_mmr
import math

# dataset 1
l = [70,80,85,90,95]
# #l = [50,100,150,200]
outDir = 'data/data1-sys-summary-sen_avg-compression-300/'
if os.path.exists(outDir):
    shutil.rmtree(outDir)
    os.makedirs(outDir)
else:
    os.makedirs(outDir)

path = "data/body/"
filens = os.listdir(path)
stops = stopwords.words('english')
for idf,filen in enumerate(filens):
    '''
    idf=0
    filen=filens[idf]
    '''
    print(idf)
    sentence, word, textlength = preprocess(path+filen,stops)
    nodes = textrank(word)
    sentences = sentence_weight(sentence,word,nodes)
    #generate_summary_bylength(sentences,word,nodes,filen, outDir,l)
    generate_summary_bycompression(sentences,filen, outDir,l, textlength)
    #generate_summary_bylength_mmr(sentences,word,nodes,filen, outDir,l,0.7)
    #generate_summary_bycompression_mmr(sentences,word,nodes,filen, outDir,l,textlength,0.9)
    
    
    
#dataset 2
# l = [70,80,90,95,98]
# #l = [50,100,150,200]
# outDir = 'data2/data2-sys-summary-sen_avg-compression-300/'
# if os.path.exists(outDir):
#     shutil.rmtree(outDir)
#     os.makedirs(outDir)
# else:
#     os.makedirs(outDir)
#
# path = "data2/body/"
# filens = os.listdir(path)
# stops = stopwords.words('english')
# for idf,filen in enumerate(filens):
#     #filen=filens[0]
#     print(idf)
#     fpath=path+filen
#     sentence, word, textlength = preprocess(fpath,stops)
#     nodes = textrank(word)
#     sentences = sentence_weight(sentence,word,nodes)
#     #generate_summary_bylength(sentences,word,nodes,filen, outDir,l)
#     generate_summary_bycompression(sentences,filen, outDir,l, textlength)
#     #generate_summary_bylength(sentences,word,nodes,filen, outDir,l)
#     #generate_summary_bylength_mmr(sentences,word,nodes,filen, outDir,l,0.9)
    