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
from summarizing import generate_summary

outDir = 'data/sys-summary/'
if os.path.exists(outDir):
    shutil.rmtree(outDir)
    os.makedirs(outDir)
else:
    os.makedirs(outDir)

path = "data/body/"
filens = os.listdir(path)
for filen in filens[0:100]:
    sentence, word = preprocess(path+filen)
    nodes = textrank(word)
    generate_summary(sentence,word,nodes,filen)
    