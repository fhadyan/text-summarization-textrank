from wordfreq import word_frequency
from wordfreq import zipf_frequency
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
from nltk.tag import StanfordNERTagger
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from itertools import chain
from nltk.stem.porter import PorterStemmer

SS_PARAMETERS_TYPE_MAP = {'definition':str, 'lemma_names':list, 
                          'examples':list,  'hypernyms':list,
                         'hyponyms': list, 'member_holonyms':list,
                         'part_holonyms':list, 'substance_holonyms':list,
                         'member_meronyms':list, 'substance_meronyms': list,
                         'part_meronyms':list, 'similar_tos':list}

def preprocess(fpath):
    file = open(fpath,"r+", encoding="utf-8")
    text = file.read()
    text=re.sub(r'\(CNN\)',' ',text)
    text=re.sub(r'(?<=[a-z])\.','\n',text)
    
    sentence = re.split(r'\n',text)
    sentence = [x for x in sentence if len(x)>1]
    
    text = re.split(r'\n',text)
    text = [x for x in text if len(x)>1]
    text = [word_tokenize(x) for x in text]
    text = [nltk.pos_tag(x) for x in text]
    text = [[[re.sub(r'[^a-zA-Z]+','',x[0]), x[1]] for x in y] for y in text]
    text = [[x for x in y if len(x[0])>1] for y in text]
    
    ner = [st.tag([x[0] for x in y if x[0] not in stops]) for y in text]
    ner = [[[x[0].lower(), text[iy][ix][1], x[1]] for ix,x in enumerate(y)] for iy,y in enumerate(ner)]
    ner = [[x for x in y if x[2]=='O'] for y in ner]

    return sentence, ner

def cwi(sentences ,words):
    word_freqs = [[[x[0], x[1],x[2], zipf_frequency(x[0], 'en')] for x in y] for y in words]
    word_freqs_sorted = [sorted(y, key=lambda x: x[3], reverse=False) for y in word_freqs]
    word_freq = word_freqs_sorted[1][0] #####
    sentence = word_freqs_sorted[1]#####
    for x in word_freqs:
        subtitute = lesk(word_freqs_sorted[1],word_freqs_sorted[1][0])
        
def synProp(synset, param):
    #synset=s #####
    #param='definition' #####
    paramType = SS_PARAMETERS_TYPE_MAP[param]
    func = 'synset.' + param
    return eval(func) if isinstance(eval(func), paramType) else eval(func)()
        
def synsetSignature(amb_word,word_freq):
    syn = wordnet.synsets(amb_word)
    synSignature = {}
    for s in syn:
        #s=syn[0] #####
        sig = []
        sigDef = synProp(s, 'definition')
        sig+=sigDef.split()
        sigExam = synProp(s,'examples')
        sig+=list(chain(*[i.split() for i in sigExam]))
        sigLemma = synProp(s, 'lemma_names')
        sig+=sigLemma
        sigHypo = synProp(s, 'hyponyms')
        sigHype = synProp(s, 'hypernyms')
        sig+=list(chain(*[i.lemma_names() for i in sigHypo+sigHype]))
        sig = [i for i in sig if i not in stops]
        sig = [wordnetlemmatizer.lemmatize(i) for i in sig]
        sig = [stemmer.stem(i) for i in sig]
        synSignature[s]=sig
    return synSignature
    
def lesk(sentence,word_freq):
    amb_word = wordnetlemmatizer.lemmatize(word_freq[0])
    if not wordnet.synsets(amb_word):
        return 0
    synSign = synsetSignature(amb_word,word_freq)
    for s in synSign:
        sMemholo = synProp(s, 'member_holonyms')
        sPartholo = synProp(s, 'part_holonyms')
        sSubholo = synProp(s, 'substance_holonyms')
        sMemmero = synProp(s, 'member_meronyms')
        sPartmero = synProp(s, 'part_meronyms')
        sSubmero = synProp(s, 'substance_meronyms')
        sSimilar = synProp(s, 'similar_tos')
        relatedSense = list(set(sMemholo+sPartholo+sSubholo+sMemmero+sPartmero+sSubmero+sSimilar))
        sig = list([x for x in chain(*[synProp(y, 'lemma_names') for y in relatedSense]) if x not in stops])
        sig = [wordnetlemmatizer.lemmatize(x) for x in sig]
        sig = [stemmer.stem(x) for x in sig]
        synSign[s]=sig
    lemmatizeSentence = [[wordnetlemmatizer.lemmatize(x[0]),x[1],x[2],x[3]] for x in sentence]
    sense = rankedSynset(lemmatizeSentence, synSign)
    return sense
    

def rankedSynset(lemmatizeSentence, synSign):
    overlaps=[]
    lemmatizeWord = [x[0] for x in lemmatizeSentence]
    for s in synSign:
        overlap = set(synSign[s]).intersection(lemmatizeWord)
        overlaps.append([len(overlap),s])
    rank = sorted(overlaps,reverse=True)    
    total = float(sum(x[0] for x in rank))
    if total==0:
        total=1
    rank = [[x[0]/total, x[1]] for x in rank]
    return rank
    
        
        
    
        
    

    
            
            
            

st = StanfordNERTagger('stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz','stanford-ner/stanford-ner.jar', encoding='utf-8')
stops = stopwords.words('english')
wordnetlemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

path='data/sys-summary/'
dirpaths = os.listdir(path)
for dirpath in dirpaths:
    dirpath = dirpaths[10] #####
    filenames = os.listdir(path+dirpath)
    for filename in filenames:
        filename = filenames[2] #####
        sentences,words = preprocess(path+dirpath+"/"+filename)
        
        