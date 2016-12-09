import PythonROUGE
import os
import numpy as np
import csv

def evaluate(sysdir, evalname):
    refdir = 'data/summary/'
    evaldir = 'data/eval/'
    evalfile = open(evaldir+evalname, 'w')
    evalcsv = csv.writer(evalfile, quoting=csv.QUOTE_ALL)
    
    dirnames = os.listdir(sysdir)
    
    fmeasure2 = []
    recall2 = []
    precision2 = []
    for newsname in dirnames:
        refsumm = refdir+newsname+'.txt'
        syssumms = os.listdir(sysdir+newsname)
        # print(newsname+':')
        evalcsv.writerow([newsname+':', 'recall', 'precision', 'f-measure'])
        tempf = []
        tempr = []
        tempp = []
        names = []
        for syssumm in syssumms:
            sys_summ = [sysdir+newsname+'/'+syssumm]
            ref_summ = [[refsumm]]
            r,p,f = PythonROUGE.PythonROUGE(sys_summ,ref_summ,ngram_order=2)
            # print(syssumm+' =  r:'+str(r)+'; p:'+str(p)+'; f:'+str(f))
            evalcsv.writerow([syssumm, str(r), str(p), str(f)])
            tempf.append(f[1])
            tempr.append(r[1])
            tempp.append(p[1])
            names.append(syssumm)
        fmeasure2.append(tempf)
        recall2.append(tempr)
        precision2.append(tempp)
        # print('\n')
        evalcsv.writerow([])
    
    f12 = np.array(fmeasure2)
    r12 = np.array(recall2)
    p12 = np.array(precision2)
    # print('\nsystem overall:')
    evalcsv.writerow(['System Overall', 'recall', 'precision', 'f-measure'])
    for i,x in enumerate(names):
        # print(str(x[:-4])+"  :" + str(np.sum(f12[:,i])/len(f12)) + '; '+ str(np.sum(r12[:,i])/len(r12)) + '; '+ str(np.sum(p12[:,i])/len(p12)) + '; ')
        evalcsv.writerow([str(x[:-4]), str(np.sum(f12[:,i])/len(f12)), str(np.sum(r12[:,i])/len(r12)), str(np.sum(p12[:,i])/len(p12))])
        

        
sysdir = 'data/sys-summary/'
evalname = 'sys1.csv'
evaluate(sysdir,evalname)