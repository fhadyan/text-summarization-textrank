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

summary_path = 'data/data1-sys-summary-sen_avg-compression-300-mmr-9/'
wsd_path = 'data/data1-sys-summary-sen_avg-compression-300-mmr-9-wsd2/'
reference_path = 'data/summary/'

analysis_path = 'data/analysis/'
afile = open(analysis_path+'analysis1-2-a.csv', 'w', newline='')
acsv = csv.writer(afile, quoting=csv.QUOTE_ALL)

wsd_files = os.listdir(wsd_path)

total_diff=0

for idf,file_dir in enumerate(wsd_files):
    '''
    print(idf+1)
    idf=0
    file_dir = summary_files[idf]
    '''
    acsv.writerow([file_dir,'---------------------------'])
    compression_files = os.listdir(wsd_path+file_dir)
    for idf2,file in enumerate(compression_files):
        '''
        idf2=2
        file=compression_files[idf2]
        '''
        acsv.writerow([file])

        summary_file_path = summary_path+file_dir+'/'+file[2:]
        reference_file_path = reference_path+file_dir+'.txt'
        rs,ps,fs = PythonROUGE.PythonROUGE([summary_file_path],[[reference_file_path]],ngram_order=2)
        wsd_file_path = wsd_path+file_dir+'/'+file
        rw,pw,fw = PythonROUGE.PythonROUGE([wsd_file_path],[[reference_file_path]],ngram_order=2)
        fd=fs[0]-fw[0]
        if(fw[0]>fs[0]):
            rouge_diff = 'up'
        elif(fw[0]<fs[0]):
            rouge_diff = 'down'
        else:
            rouge_diff = 'same'
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
            acsv.writerow([x[0],x[1]])
    acsv.writerow([])
acsv.writerow(['total difference',total_diff])
afile.close()







