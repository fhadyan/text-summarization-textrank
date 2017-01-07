import PythonROUGE
import os
import numpy as np
import csv

def evaluate(sysdir, evalname,refdir,evaldir):
    #refdir = 'data/summary/'
    #evaldir = 'data/eval/'
    evalfile = open(evaldir+evalname, 'w', newline='')
    evalcsv = csv.writer(evalfile, quoting=csv.QUOTE_ALL)
    
    dirnames = os.listdir(sysdir)
    systemsummary = dict()
    referencesummary = dict()
    for newsname in dirnames:
        #newsname = dirnames[0]
        refsumm = refdir+newsname+'.txt'
        syssumms = os.listdir(sysdir+newsname)
        for syssumm in syssumms:
            #syssumm = syssumms[0]
            sys_summ = sysdir+newsname+'/'+syssumm
            ref_summ = refsumm
            try:
                systemsummary[syssumm] += sys_summ+'|'
            except:
                systemsummary[syssumm] = sys_summ+'|'
            try:
                referencesummary[syssumm] += refsumm+'|'
            except:
                referencesummary[syssumm] = refsumm+'|'
    for x in systemsummary:
        systemsummary[x]=systemsummary[x].split('|')
        systemsummary[x]=[y for y in systemsummary[x] if len(y)>0]
        referencesummary[x]=referencesummary[x].split('|')
        referencesummary[x]=[[y] for y in referencesummary[x] if len(y)>0]
    ev=[]
    for idx,x in enumerate(systemsummary):
        '''
        x=systemsummary[0]
        '''
        r,p,f = PythonROUGE.PythonROUGE(systemsummary[x],referencesummary[x],ngram_order=2)
        ev.append([x,r,p,f])
    for i in range(0,len(ev)):
        try:
            #ev[i].append(int(ev[i][0][:-4]))
            ev[i].append(ev[i][0][:-4])
        except:
            #ev[i].append(int(ev[i][0][:-5]))
            ev[i].append(ev[i][0][:-5])
    ev.sort(key=lambda x: x[4])
    range_rouge=['1','2','L','W-1.2'] 
    for i in range(0,len(ev[0][1])):
        print(i)
        evalcsv.writerow(['Rouge '+range_rouge[i]])
        evalcsv.writerow([' ','Recall','Preission','F-measure'])
        for j in range(0,len(ev)):
            evalcsv.writerow([ev[j][0],ev[j][1][i],ev[j][2][i],ev[j][3][i]])
        evalcsv.writerow([])
        #print('--------------'+x+'--------------')
        #print(r)
        #print(p)
        #print(f)
      

if __name__ == "__main__": 
    sysdir = 'data/data1-sys-summary-sen_avg-compression-300-mmr-9-wsd2/'
    refdir = 'data/summary/'
    evalname = 'data1-sys-summary-sen_avg-compression-300-mmr-9-wsd2.csv'
    evaldir = 'data/eval/'
    evaluate(sysdir,evalname,refdir,evaldir)
