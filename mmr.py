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
from nltk.stem.porter import PorterStemmer
from collections import Counter

def pre(t1):
    stops = stopwords.words('english')
    stemmer = PorterStemmer()
    t=re.sub(r'[^a-zA-Z0-9\s]',' ',t1)
    t=t.split(' ')
    t=[tx for tx in t if len(tx)>2]
    t=[tx for tx in t if tx not in stops]
    t=[stemmer.stem(tx) for tx in t]
    t=' '.join(t)
    return t
    

def calcidf(sentences):
    stops = stopwords.words('english')
    nd=len(sentences)
    idf={}
    idfn=0
    ws={}
    for x in sentences:
        '''
        x=sentences[0]
        '''
        t=pre(x[1])
        wt=[]
        for w in t:
            #w=t[0]
            if ws.get(w, 0) == 0:
                ws[w]=1
                wt.append(w)
            elif ws.get(w, 0) != 0 and w not in wt:
                ws[w]+=1
    
    for w in ws:
        i = ws[w]
        try:
            idfn=math.log10(float(nd)/i)
        except:
            idfn=0
        idf[w]=idfn
    return idf

def tfidf(sentences):
    tfidf={}
    stops = stopwords.words('english')
    tf={}
    for x in sentences:
        '''
        x=sentences[0]
        '''
        # tf
        t=pre(x[1])
        for w in t:
            #w=t[0]
            if tf.get(w,0)!=0:
                tf[w]+=1
            else:
                tf[w]=1
        
    # idf
    nd=len(sentences)
    idf={}
    idfn=0
    ws={}
    for x in sentences:
        '''
        x=sentences[0]
        '''
        t=pre(x[1])
        wt=[]
        for w in t:
            #w=t[0]
            if ws.get(w, 0) == 0:
                ws[w]=1
                wt.append(w)
            elif ws.get(w, 0) != 0 and w not in wt:
                ws[w]+=1
    
    for w in ws:
        i = ws[w]
        try:
            idfn=math.log10(float(nd)/i)
        except:
            idfn=0
        idf[w]=idfn
    tfidf_=0
    for w in tf:
        tfidf_=tf[w]*idf[w]
        tfidf[w]=tfidf_
    return tfidf
            

def calctfidf(t,idf):
    tfidf_=0
    tf={}
    t_vec={}
    for w in t:
        if tf.get(w,0)!=0:
            tf[w]+=1
        else:
            tf[w]=1
    for w in tf:
        if idf.get(w,0)!=0:
            tfidf_=tf[w]*idf[w]
            t_vec[w]=tfidf_
        else:
            t_vec[w]=0
    return t_vec

def text_to_vector(text):
    words = re.compile(r'\w+').findall(text)
    return Counter(words)
    
def cosineSimilarity2(t1,t2):
    t1_vec = {}
    t2_vec = {}
    t=pre(t1)
    t1_vec = text_to_vector(t)
    t=pre(t2)
    t2_vec = text_to_vector(t)
    
    inter = set(t1_vec.keys()) & set(t2_vec.keys())
    num = sum([t1_vec[x] * t2_vec[x] for x in inter])
    
    sum1 = sum([t1_vec[x]**2 for x in t1_vec.keys()])
    sum2 = sum([t2_vec[x]**2 for x in t2_vec.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(num) / denominator
    

def cosineSimilarity(t1,t2,idf):
    t1_vec = {}
    t2_vec = {}
    t=pre(t1)
    t1_vec = calctfidf(t,idf)
    t=pre(t2)
    t2_vec = calctfidf(t,idf)
    
    inter = set(t1_vec.keys()) & set(t2_vec.keys())
    num = sum([t1_vec[x] * t2_vec[x] for x in inter])
    
    sum1 = sum([t1_vec[x]**2 for x in t1_vec.keys()])
    sum2 = sum([t2_vec[x]**2 for x in t2_vec.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(num) / denominator

def mmrReorder(sentences,n):
    r_s = [x for x in sentences if x[3]==0]
    for idi,di in enumerate(r_s):
        #print(idi)
        '''
        idi=0
        di = r_s[0]
        n=0.7
        '''
        #di_score=cosineSimilarity(di[1],q,idf)
        di_score=di[4]
        dj_score_s=[]
        dj_score_s.append(0)
        s = [x for x in sentences if x[3]==1]
        for dj in s:
            '''
            dj=s[0]
            '''
            dj_score=cosineSimilarity2(dj[1],di[1])
            dj_score_s.append(dj_score)
        dj_score_m=max(dj_score_s)
        mmrscore=(n*float(di_score))-((1-n)*dj_score_m)
        r_s[idi][2]=mmrscore
    new_sentences = [x for x in sentences if x[3]!=0]
    new_sentences.extend(r_s)
    return new_sentences
        
        


if __name__ == "__main__":
    print('ok')