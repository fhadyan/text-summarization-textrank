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
import math

def preprocess(fpath,stops):
    # fpath=path+filens[0]
    file = open(fpath,"r+", encoding="utf-8")
    text = file.read()
    text=re.sub(r'\(CNN\)',' ',text)
    text=re.sub(r'[\"]',' ',text)
    text=re.sub(r'[^0-9a-zA-Z\s]{2,}',' ',text)
    text=re.sub(r'(?<=[a-z])\.','\n',text) #### preprocession 2
    text=re.sub(r'(?<=\s)\.','\n',text) #### preprocession 2
    #text=re.sub(r'[\s]{2,}',' ',text)
    
    textlength = len(re.split(r'\s',text))
    
    sentence = re.split(r'\n',text)
    sentence = [re.sub(r'[\s]{2,}',' ',x) for x in sentence]
    sentence = [x for x in sentence if len(x)>5]
                
    text=text.lower()
    text = re.sub("[^0-9a-zA-Z\s]+", "", text)
    word = re.split(r'\n',text)
    word = [re.sub(r'[\s]{2,}',' ',x) for x in word]
    word = [x for x in word if len(x)>5]
    word = [nltk.word_tokenize(x) for x in word if len(x)>1]
    word = [[x for x in y if len(x)>2] for y in word]
    word = [[x for x in y if x not in stops] for y in word]

    return sentence, word, textlength


if __name__ == "__main__":
    path = "data/body/"
    filens = os.listdir(path)
    stops = stopwords.words('english')
    sentence, word = preprocess(path+filens[0])