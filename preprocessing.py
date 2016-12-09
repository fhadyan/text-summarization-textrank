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

def preprocess(fpath):
    # fpath=path+filens[0]
    file = open(fpath,"r+", encoding="utf-8")
    text = file.read()
    text=re.sub(r'\(CNN\)',' ',text)
    text=re.sub(r'[^0-9a-zA-Z\s]{2,}',' ',text)
    text=re.sub(r'[\s]{2,}',' ',text)
    
    sentence = re.split(r'\n',text)
    sentence = [x for x in sentence if len(x)>1]
                
    text=text.lower()
    text = re.sub("[^0-9a-zA-Z\s]+", "", text)
    word = re.split(r'\n',text)
    word = [nltk.word_tokenize(x) for x in word if len(x)>1]
    word = [[x for x in y if x not in stops] for y in word]

    return sentence, word



path = "data/body/"
filens = os.listdir(path)
stops = stopwords.words('english')
sentence, word = preprocess(path+filens[0])