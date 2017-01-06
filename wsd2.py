from string import punctuation
from nltk import word_tokenize
from nltk.corpus import wordnet as wn
from nltk.corpus import brown, stopwords
from pywsd.lesk import adapted_lesk
from pywsd.similarity import max_similarity
from pywsd.utils import lemmatize
from pywsd.allwords_wsd import disambiguate

import os
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from wordfreq import word_frequency
from wordfreq import zipf_frequency
import copy

def wsd(summpath, wsdpath):
    summpathlist = os.listdir(summpath)
    for idp,dirpath in enumerate(summpathlist):
        print(idp)
        '''
        idp=0
        dirpath=summpathlist[idp]
        '''
        comppath = summpath+dirpath+'/'
        comppathlist = os.listdir(comppath)
        for idc,compathdir in enumerate(comppathlist):
            '''
            idc=0
            compathdir=comppathlist[idc]
            '''
            fpath = comppath+compathdir
            f = open(fpath,'r+',encoding='utf-8')
            text = f.readlines()
            text = [re.sub(r'^\s+|\s+$','',x) for x in text if len(x)>3]
            textwsd = []
            for ids,sentence in enumerate(text):
                '''
                ids=0
                sentence=text[ids]
                '''
                sentencewsd=copy.deepcopy(sentence)
                ambiguity = disambiguate(sentence, adapted_lesk, keepLemmas=True)
                for idy,syn in enumerate(ambiguity):
                    '''
                    idy=3
                    syn=ambiguity[idy]
                    '''
                    if syn[2] is not None:
                        syn_lemma = syn[2].lemma_names()
                        syn_lemma = [[zipf_frequency(x, 'en'),x] for x in syn_lemma ]
                        syn_lemma = sorted(syn_lemma , reverse=True)
                        sentencewsd = re.sub(r''+syn[0],syn_lemma[0][1],sentencewsd )
                textwsd.append(sentencewsd)
            outDirectory = wsdpath+dirpath+'/'
            if not os.path.exists(outDirectory):
                os.makedirs(outDirectory)
            fout = open(os.path.join(outDirectory, compathdir),'w',encoding='utf-8')
            fout.writelines(textwsd)
            fout.close()
            
                        
                    
            
    
if __name__ == "__main__":
    summpath = "data/data1-sys-summary-sen_avg-compression-10/"
    wsdpath = "data/data1-sys-summary-sen_avg-compression-10-wsd2/"
    wsd(summpath,wsdpath)