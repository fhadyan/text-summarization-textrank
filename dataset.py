import os
from platform import node
import copy
import nltk
import re
import numpy as np
from nltk.corpus import stopwords
from pyrouge import Rouge155
from pprint import pprint

dirn_ori = "data/original/"
dirn_body = "data/body/"
dirn_summ = "data/summary/"
filesn_out = os.listdir(dirn_ori)

for filen in filesn_out:
    print(filen)
    
    f=open(dirn_ori+filen, "r+", encoding="UTF-8")
    text=f.readlines()
    text=[x for x in text if len(x)>1]
    summ_start = [i for i,x in enumerate(text) if x == "@highlight\n"]
    text_b=[x for x in text[0:summ_start[0]]]
    text_s=[x for x in text[summ_start[0]:]]
    text_s=[x for x in text_s if x != "@highlight\n"]

    fb=open(dirn_body+filen[:-5]+"txt", "w+", encoding="UTF-8")
    fb.writelines(text_b)
    fb.flush()
    fb.close()
    
    fs=open(dirn_summ+filen[:-5]+"txt", "w+", encoding="UTF-8")
    fs.writelines(text_s)
    fs.flush()
    fs.close()