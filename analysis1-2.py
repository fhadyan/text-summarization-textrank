import PythonROUGE
from platform import node
import copy
import nltk
import re
import numpy as np
from pyrouge import Rouge155
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
from nltk.stem.porter import *

stemmer = PorterStemmer()

summary_path = 'data/data1-sys-summary-sen_avg-compression-300-mmr-9/'
wsd_path = 'data/data1-sys-summary-sen_avg-compression-300-mmr-9-wsd2/'
reference_path = 'data/summary/'

analysis_path = 'data/analysis/'
afile = open(analysis_path+'analysis2-data1-50.csv', 'w', newline='')
acsv = csv.writer(afile, quoting=csv.QUOTE_ALL)

wsd_files = os.listdir(wsd_path)
total_diff=0

##### rouge store
rouge_store = {}

##### create dictionary
rouge_dict = {}
rouge_f = os.listdir(wsd_path+wsd_files[0])
for x in rouge_f:
    rouge_dict[x]=[0,0,0,0,0,0,0]
    '''
    0: rouge up
    1: rouge same
    2: rouge down
    3: dig+ wsd+
    4: dig+ wsd-
    5: dig- wsd-
    6: dig- wsd+
    '''

for idf,file_dir in enumerate(wsd_files[0:50]):
    '''
    print(idf+1)
    idf=3
    file_dir = wsd_files[idf]
    '''
    print(idf)
    ref_file = open(reference_path+file_dir+'.txt','r+',encoding='utf-8')
    ref_text = ref_file.read()
    ref_text = re.sub(r'\.|\s',' ',ref_text)
    ref_text = ref_text.split(' ')
    ref_text = [stemmer.stem(x) for x in ref_text]
    ref_text = set(ref_text)
    
    
    acsv.writerow([file_dir,'---------------------------'])
    compression_files = os.listdir(wsd_path+file_dir)
    for idf2,file in enumerate(compression_files):
        '''
        idf2=0
        file=compression_files[idf2]
        '''
        acsv.writerow([file])

        summary_file_path = summary_path+file_dir+'/'+file[2:]
        reference_file_path = reference_path+file_dir+'.txt'
        try:
            rs=rouge_store[summary_file_path][0]
            ps=rouge_store[summary_file_path][1]
            fs=rouge_store[summary_file_path][2]
        except:
            rs,ps,fs = PythonROUGE.PythonROUGE([summary_file_path],[[reference_file_path]],ngram_order=2)
            rouge_store[summary_file_path]=[rs,ps,fs]
        wsd_file_path = wsd_path+file_dir+'/'+file
        rw,pw,fw = PythonROUGE.PythonROUGE([wsd_file_path],[[reference_file_path]],ngram_order=2)
        fd=fs[0]-fw[0]
        if(fw[0]>fs[0]):
            rouge_diff = 'up'
            rouge_dict[file][0]+=1
        elif(fw[0]<fs[0]):
            rouge_diff = 'down'
            rouge_dict[file][2]+=1
        else:
            rouge_diff = 'same'
            rouge_dict[file][1]+=1
        acsv.writerow(['rouge',fs[0],fw[0],fd,rouge_diff])

        summary_file = open(summary_file_path, 'r+')
        summary_text = summary_file.read()
        summary_text = re.sub(r'\.',' ',summary_text)
        summary_text = re.sub(r'\s{2}',' ',summary_text)
        summary_text = summary_text.split(' ')
        summary_text = [x for x in summary_text if len(x)>0]

        wsd_file = open(wsd_file_path, 'r+')
        wsd_text = wsd_file.read()
        wsd_text = re.sub(r'\.',' ',wsd_text)
        wsd_text = re.sub(r'\s{2}',' ',wsd_text)
        wsd_text = wsd_text.split(' ')
        wsd_text = [x for x in wsd_text if len(x)>0]
        diff = [[x,wsd_text[idx]] for idx,x in enumerate(summary_text) if x!=wsd_text[idx]]
        total_diff+=len(diff)
        acsv.writerow(['word',len(diff)])
        for x in diff:
            '''
            x=diff[0]
            '''
            dig_ref = True if stemmer.stem(x[0]) in ref_text else False
            wsd_ref = True if stemmer.stem(x[1]) in ref_text else False
            if dig_ref and wsd_ref:
                word_diff='dig+ wsd+'
                rouge_dict[file][3]+=1
            elif dig_ref and not wsd_ref:
                word_diff='dig+ wsd-'
                rouge_dict[file][4]+=1
            elif not dig_ref and not wsd_ref:
                word_diff='dig- wsd-'
                rouge_dict[file][5]+=1
            elif not dig_ref and wsd_ref:
                word_diff='dig- wsd+'
                rouge_dict[file][6]+=1
            acsv.writerow([x[0],x[1],word_diff])
            
            
    acsv.writerow([])
acsv.writerow(['total difference',total_diff])
afile.close()

rfile = open(analysis_path+'analysis2-data1-50-summary.csv', 'w', newline='')
rcsv = csv.writer(rfile, quoting=csv.QUOTE_ALL)
for x in rouge_dict:
    #print(rouge_dict[x])
    y=[x]
    y.extend(rouge_dict[x])
    rcsv.writerow(y)
rfile.close()








