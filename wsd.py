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
    text=re.sub(r'[.]{2,}', '.',text)
    text=re.sub(r'(?<=[a-z])\.','\n',text)
    
    sentence = re.split(r'\n',text)
    sentence = [x for x in sentence if len(x)>1]
    
    text = re.split(r'\n',text)
    text = [x for x in text if len(x)>1]
    text = [word_tokenize(x) for x in text]
    text = [nltk.pos_tag(x) for x in text]
    text = [[[re.sub(r'[^a-zA-Z]+','',x[0]), x[1]] for x in y] for y in text]
    text = [[x for x in y if len(x[0])>1] for y in text]
    text = [x for x in text if x!=[]]
    
    # ner = [[[y[0], y[1], 'O'] for y in x] for x in text]
    # ner = [st.tag([x[0] for x in y if x[0] not in stops]) for y in text]
    # ner = [[[x[0].lower(), text[iy][ix][1], x[1]] for ix,x in enumerate(y)] for iy,y in enumerate(ner)]
    # ner = [[x for x in y if x[2]=='O'] for y in ner]
    # ner = [x for x in ner if x!=[]]
    ner = [st.tag([x[0] for x in y if x[0] not in stops]) for y in text]
    ner = [[[x[0].lower(), text[iy][ix][1], x[1]] for ix,x in enumerate(y)] for iy,y in enumerate(ner)]
    ner = [[x for x in y if x[2]=='O'] for y in ner]
    ner = [x for x in ner if x!=[]]

    return sentence, ner

def cwi(sentences ,words):
    word_freqs = [[[x[0], x[1],x[2], zipf_frequency(x[0], 'en')] for x in y] for y in words]
    word_freqs_sorted = [sorted(y, key=lambda x: x[3], reverse=False) for y in word_freqs]
    #word_freq = word_freqs_sorted[1][0] #####
    #sentence = word_freqs_sorted[0]#####
    for i,wf_sort in enumerate(word_freqs_sorted):
        #subtitute = lesk(word_freqs_sorted[i],word_freqs_sorted[i][0]) ######
        #i=3
        #wf_sort=word_freqs_sorted[i]
        #subtitute = lesk(sentences[i],wf_sort[0])
        subtitute = lesk(sentences[i],wf_sort[0])
        #subtitute = lesk(wf_sort,wf_sort[0])
        if subtitute==0:
            continue
        if not subtitute[0][0]>0:
            continue
        #synset = subtitute[0][1].name().split('.')[1:3]
        #subWord = [x[1] for x in subtitute]
        #subWord = [x.name().split('.') for x in subWord]
        #subWord = [x[0] for x in subWord if x[1:3] == synset]
        
        synset = subtitute[0][1]
        subWord = synset.lemma_names()

        subWord = [[zipf_frequency(x, 'en'),x] for x in subWord]
        subWord = sorted(subWord, reverse=True)
        sentences[i] = re.sub(wf_sort[0][0], subWord[0][1], sentences[i])
    return sentences

def cwi2(sentences ,words, zipf_freq):
    word_freqs = [[[x[0], x[1],x[2], zipf_frequency(x[0], 'en')] for x in y] for y in words]
    for i,wf_sort in enumerate(word_freqs):
        ''' 
        i = 0
        wf_sort = word_freqs[i]
        '''
        for ambg_word in wf_sort:
            '''
            ambg_word = wf_sort[0]
            '''
            if(ambg_word[3]<zipf_freq):
                subtitute = lesk(sentences[i],ambg_word)
                #subtitute = lesk(wf_sort,wf_sort[0])
                if subtitute==0:
                    continue
                if not subtitute[0][0]>0:
                    continue
                synset = subtitute[0][1]
                subWord = synset.lemma_names()
        
                subWord = [[zipf_frequency(x, 'en'),x] for x in subWord]
                subWord = sorted(subWord, reverse=True)
                #sentences[i] = re.sub(wf_sort[0][0], subWord[0][1], sentences[i])
                sentences[i] = re.sub(ambg_word[0], subWord[0][1], sentences[i])
    return sentences
        
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
        #if(s.pos()!=word_freq[1]):
        #    continue
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
        sig = [stemmer.stem(i) for  i in sig]
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
        synSign[s]+=sig
    #lemmatizeSentence = [[wordnetlemmatizer.lemmatize(x[0]),x[1],x[2],x[3]] for x in sentence]
    sen=sentence.split(' ')
    lemmatizeSentence = [wordnetlemmatizer.lemmatize(x) for x in sen if len(x)>0]
    sense = rankedSynset(lemmatizeSentence, synSign)
    return sense
    

def rankedSynset(lemmatizeSentence, synSign):
    overlaps=[]
    #lemmatizeWord = [x[0] for x in lemmatizeSentence]
    lemmatizeWord = lemmatizeSentence
    for s in synSign:
        overlap = set(synSign[s]).intersection(lemmatizeWord)
        overlaps.append([len(overlap),s])
    total = float(sum(x[0] for x in overlaps))
    if total==0:
        total=1
        rank = overlaps
    else:
        rank = sorted(overlaps,reverse=True)        
    rank = [[x[0]/total, x[1]] for x in rank]
    return rank

st = StanfordNERTagger('stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz','stanford-ner/stanford-ner.jar', encoding='utf-8')
stops = stopwords.words('english')
wordnetlemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

path='data2/data2-sys-summary-sen_avg-compression-300/'
outPath='data2/data2-sys-summary-sen_avg-compression-300-wsd-cwi2/'
if os.path.exists(outPath):
    shutil.rmtree(outPath)
    os.makedirs(outPath)
else:
    os.makedirs(outPath)
dirpaths = os.listdir(path)
z=[8,7,6,5,4,3,2,1]
for idd,dirpath in enumerate(dirpaths):
    print(idd)
    #dirpath = dirpaths[0] #####
    # if not os.path.exists(outPath+dirpath):
    #         os.makedirs(outPath+dirpath)
    filenames = os.listdir(path+dirpath)
    for idf,filename in enumerate(filenames):
        #filename = filenames[0] #####
        fpath = path+dirpath+"/"+filename
        sentences,words = preprocess(fpath)
        for x in z:
            #zipf_freq = 4
            zipf_freq = x
            #text=cwi(sentences,words)
            text=cwi2(sentences,words,zipf_freq)
            opath = outPath+str(x)+"/"+dirpath
            if not os.path.exists(opath):
                os.makedirs(opath)
            f = open(opath+"/"+filename, "w", encoding="utf-8")
            output=''
            for x in text:
                output+=x+'.'
            f.write(output)
            f.close()
        