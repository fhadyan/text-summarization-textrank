import PythonROUGE
from platform import node
import copy
import nltk
import re
import numpy as np
import PythonROUGE
from pprint import pprint
from numpy import genfromtxt
import csv
import os
import shutil
from nltk.corpus import stopwords
from preprocessing import preprocess
from textrank import textrank
from summarizing import generate_summary_bylength
from summarizing import generate_summary_bycompression
from summarizing import sentence_weight
from summarizing import generate_summary_bylength_mmr
from summarizing import generate_summary_bycompression_mmr

##### rata rata panjang ringkasan
dir = 'data/summary/'
fn = os.listdir(dir)
n=[]
for x in fn:
    f=open(dir+x, 'r+', encoding='utf-8')
    fr=f.read()
    n.append(len(fr.split(' ')))
avg = sum(n)/len(n)
    
##### rata rata panjang berita
dir = 'data/body/'
fn = os.listdir(dir)
n=[]
for x in fn:
    f=open(dir+x, 'r+', encoding='utf-8')
    fr=f.read()
    n.append(len(fr.split(' ')))
avg = sum(n)/len(n)

##### rata rata panjang ringkasan 
dir = 'data2/summary/'
fn = os.listdir(dir)
n=[]
for x in fn:
    f=open(dir+x, 'r+', encoding='utf-8')
    fr=f.read()
    n.append(len(fr.split(' ')))
avg = sum(n)/len(n)

##### rata rata panjang data2
dir = 'data2/body/'
fn = os.listdir(dir)
n=[]
for x in fn:
    f=open(dir+x, 'r+', encoding='utf-8')
    fr=f.read()
    n.append(len(fr.split(' ')))
avg = sum(n)/len(n)

##### testing rouge
sys='data-explore/sys.txt'
ref='data-explore/ref.txt'
r,p,f = PythonROUGE.PythonROUGE([sys],[[ref]],ngram_order=2)
print(f[0])

##### sorting nodes
sorted(nodes, key=lambda x:-x[2])

# creating summary
l = [100]
outDir = 'data-explore/summary/'
path = "data/body/"
filens = os.listdir(path)
filen = filens[94]
stops = stopwords.words('english')
fpath = path+filen
fpath = 'data-explore/sys.txt'
sentence, word, textlength = preprocess(fpath,stops)
nodes = textrank(word)
sentences = sentence_weight(sentence,word,nodes)
generate_summary_bylength(sentences,word,nodes,filen, outDir,l)
#generate_summary_bycompression(sentences,filen, outDir,l, textlength)
#generate_summary_bylength_mmr(sentences,word,nodes,filen, outDir,l,0.9)
#generate_summary_bycompression_mmr(sentences,word,nodes,filen, outDir,l,textlength,0.9)

sys='data-explore/summary/'+filen[:-4]+'/'+str(l[0])+'.txt'
ref='data/summary/'+filen
r,p,f = PythonROUGE.PythonROUGE([sys],[[ref]],ngram_order=2)
print(f[0])

##### calculating sentence weight
x=word[15]
sums = 0
j=0
sw=0
for y in x:
    sums += nodes[nodes[:,0]==y,2][0]
    j+=1
if j>0:
    sw=sums/j ### average based
print(sw)