import os
from platform import node
import copy
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from pyrouge import Rouge155
from pprint import pprint
import xml.etree.ElementTree as ET

path = 'data2/original/'
bodypath = 'data2/body/'
summarypath = 'data2/summary/'
filenames = os.listdir(path)

for idx,x in enumerate(filenames[:-1]):
    #x=filenames[17]
    #print(idx)
    f = ET.parse(path+x)
    root = f.getroot()
    
    abstract= root.findall('ABSTRACT')
    if(abstract[0].findall('P')==[]):
        continue
    summ = abstract[0].findall('P')[0]
    summ = summ.text
    summ = re.sub(r'\\n','',summ)
    summ = re.sub(r'\s{2}',' ',summ)
    fs=open(summarypath+x[:-3]+"txt", "w+", encoding="utf-8")
    fs.writelines(summ)
    
    text=''
    fff=True
    for y in root.iter('DIV'):
        #y=root.findall('BODY')[0].findall('DIV')[2]
        #print(y.attrib['ID'])
        h=y.findall('HEADER')[0]
        header=h.text
        header=re.sub(r'\s','',header)
        header=str(header).lower()
        if(header=='footnotes' or header=='acknowledgements' or header=='reference' or header=='references' or header=='bibliography'):
            #print('ok')
            fff=False
            break
        
        for z in y.findall('P'):
            #z=y.findall('P')[1]
            t=ET.tostring(z,encoding='utf-8', method='xml')
            t=str(t)[2:-1]
            t=re.sub(r'<[^>]*>',' ',t)
            t=re.sub(r'\\n',' ',t)
            t=re.sub(r'[\s]{2,}',' ',t)
            t=re.sub(r'[.](?=\s)','.\\n',t)
            t=re.sub(r'(?<=[a-z])[.](?=[A-Z])',r'.\\n',t)
            t=re.sub(r'(?<=\s)[.](?=[A-Z])',r'.\\n',t)
            #t=re.sub(r'[\s]{2,}',' ',t)
            #t=re.sub(r'(?<=)[(.)]','.\n',t)
            text+=' '+t
    
    fb=open(bodypath+x[:-3]+"txt", "w+", encoding="utf-8")
    fb.write(text)
    if(fff):
        print(idx)