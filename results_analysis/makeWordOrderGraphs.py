#from pylab import *
#import pylab
from matplotlib import rcParams
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
markerlist = ['o','s','^','h','d','x','v','>','<','p','*',',']

def doAdj(probdir,detqnadjplot):
    probfile = open(probdir+"/adj-prob-file.dat","r").readlines()
#    pp = PdfPages(probdir+"/adjruleplot.pdf")
#    fig = plt.figure()
#    ax1 = fig.add_subplot(111)
    ax2 = detqnadjplot#.add_subplot(111)

    npn = []
    nnp = []
    for i in range(3,min(len(probfile)-2,3002)):
        line = probfile[i]
        if line.find(":")!=-1: continue
        prob1 = float(line.strip().rstrip().split()[0].strip().rstrip())
        npn.append(prob1)
        prob2 = float(line.strip().rstrip().split()[1].strip().rstrip())
        nnp.append(prob2)
#    ax1.plot(range(len(probfile)-2-2),npn,label="NP/N")
    ax2.plot(range(len(npn)),npn,label="Adj first")
#    ax1.plot(range(len(probfile)-2-2),nnp,label="N NP")
    ax2.plot(range(len(npn)),nnp,label="Adj last")
    pass
#    box = ax1.get_position()
#    ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
#    leg = ax1.legend(loc='center left',bbox_to_anchor=(1,0.5))
#    pp.savefig()
#    pp.close()


def doDet(probdir,detqnadjplot):
    probfile = open(probdir+"/det-prob-file.dat","r").readlines()
#    pp = PdfPages(probdir+"/detruleplot.pdf")
#    fig = plt.figure()
#    ax1 = fig.add_subplot(111)
    ax2 = detqnadjplot#.add_subplot(111)
    npn = []
    nnp = []
    for i in range(3,min(len(probfile)-2,3002)):
        line = probfile[i]
        if line.find(":")!=-1: continue
        prob1 = float(line.strip().rstrip().split()[0].strip().rstrip())
        npn.append(prob1)
        prob2 = float(line.strip().rstrip().split()[1].strip().rstrip())
        nnp.append(prob2)
#    ax1.plot(range(len(probfile)-2-2),npn,label="NP/N")
    ax2.plot(range(len(npn)),npn,label="Det first")
#    ax1.plot(range(len(probfile)-2-2),nnp,label="$NP  N$")
#    ax2.plot(range(len(npn)),nnp,label="Det last")

    pass
#    box = ax1.get_position()
#    ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
#    leg = ax1.legend(loc='center left',bbox_to_anchor=(1,0.5))
#    pp.savefig()
#    pp.close()


def doQuant(probdir,detqnadjplot):
    probfile = open(probdir+"/qn-prob-file.dat","r").readlines()
#   pp = PdfPages(probdir+"/qnruleplot.pdf")
#   fig = plt.figure()
#    ax1 = fig.add_subplot(111)
    ax2 = detqnadjplot#.add_subplot(111)
    npn = []
    nnp = []
    for i in range(3,min(len(probfile)-2,3002)):
        line = probfile[i]
        if line.find(":")!=-1: continue
        prob1 = float(line.strip().rstrip().split()[0].strip().rstrip())
        npn.append(prob1)
        prob2 = float(line.strip().rstrip().split()[1].strip().rstrip())
        nnp.append(prob2)
#    ax1.plot(range(len(probfile)-2-2),npn,label="NP/N")
    ax2.plot(range(len(npn)),npn,label="Quant first")
#    ax1.plot(range(len(probfile)-2-2),nnp,label="$NP  N$")
    ax2.plot(range(len(nnp)),nnp,label="Quant last")

    pass
#    box = ax1.get_position()
#    ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
#    leg = ax1.legend(loc='center left',bbox_to_anchor=(1,0.5))
#    pp.savefig()
#    pp.close()







def doIntrans(probdir):
    probfile = open(probdir+"/sv-prob-file.dat","r").readlines()
    pp = PdfPages(probdir+"/svruleplot.pdf")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    npn = []
    nnp = []
    for i in range(3,min(len(probfile)-2,3002)):
        line = probfile[i]
        if line.find(":")!=-1: continue
        prob1 = float(line.strip().rstrip().split()[0].strip().rstrip())
        npn.append(prob1)
        prob2 = float(line.strip().rstrip().split()[1].strip().rstrip())
        nnp.append(prob2)
    ax1.plot(range(len(probfile)-2-2),npn,label="SV")

    ax1.plot(range(len(probfile)-2-2),nnp,label="VS")


    pass
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    leg = ax1.legend(loc='center left',bbox_to_anchor=(1,0.5))
    pp.savefig()
    pp.close()


def doTrans(probdir,ax1,numreps,marker):
    probfile = open(probdir+"/svo-prob-file.dat","r").readlines()

    
    svo = []
    ovs = []
    
    # ((S/NP)/NP) [0, 1, 2] vso
    # ((S/NP)\\NP) [0, 1, 2] svo
    # ((S\\NP)/NP) [0, 1, 2] ovs
    # ((S\\NP)\\NP) [0, 1, 2] sov
    # ((S/NP)/NP) [1, 0, 2] vos
    # ((S/NP)\\NP) [1, 0, 2] ovs
    # ((S\\NP)/NP) [1, 0, 2] svo
    # ((S\\NP)\\NP) [1, 0, 2] osv
    
    verborders = ["vso", "svo","ovs", "sov","vos", "ovs","svo", "osv" ]
    tvcats = ["","","","","","","",""] # fix

    vso = []
    svo = []
    ovs = []
    sov = []
    vos = []
    osv = []

    for i in range(9,min(3002,len(probfile)-2)):
        line = probfile[i]
        if line.find(":")!=-1: continue
        
        prob1 = float(line.strip().rstrip().split()[0].strip().rstrip())
        vso.append(prob1)
        prob2 = float(line.strip().rstrip().split()[1].strip().rstrip())
        svo.append(prob2)
        prob3 = float(line.strip().rstrip().split()[2].strip().rstrip())
        ovs.append(prob3)
        prob4 = float(line.strip().rstrip().split()[3].strip().rstrip())
        sov.append(prob4)
        prob5 = float(line.strip().rstrip().split()[4].strip().rstrip())
        vos.append(prob5)
        prob6 = float(line.strip().rstrip().split()[5].strip().rstrip())
        ovs[-1]+=prob6
        prob7 = float(line.strip().rstrip().split()[6].strip().rstrip())
        svo[-1]+=prob7
        prob8 = float(line.strip().rstrip().split()[7].strip().rstrip())
        osv.append(prob8)

#    verborders = ["vso", "svo","ovs", "sov","vos", "ovs","svo", "osv" ]
    ax1.plot(range(len(svo)),vso,label="vso, ~~$|{m}|="+str(numreps)+"$",marker=markerlist[marker],markevery=30)
    ax1.plot(range(len(svo)),svo,label="svo, ~~$|{m}|="+str(numreps)+"$",marker=markerlist[marker+1],markevery=30)
    ax1.plot(range(len(svo)),ovs,label="ovs, ~~$|{m}|="+str(numreps)+"$",marker=markerlist[marker+2],markevery=30)
    ax1.plot(range(len(svo)),sov,label="sov, ~~$|{m}|="+str(numreps)+"$",marker=markerlist[marker+3],markevery=30)
    ax1.plot(range(len(svo)),vos,label="vos, ~~$|{m}|="+str(numreps)+"$",marker=markerlist[marker+4],markevery=30)
    ax1.plot(range(len(svo)),osv,label="osv, ~~$|{m}|="+str(numreps)+"$",marker=markerlist[marker+5],markevery=30)


    pass



probdir = "newRepProbsN+1_bigSat_1W"
ppdqa = PdfPages(probdir+"/detqnadjplot_1rep.pdf")
figpdqafig = plt.figure(figsize=(rcParams['figure.figsize'][0],rcParams['figure.figsize'][1]))
figpdqa2 = figpdqafig.add_subplot(111)
plt.title("Unambiguous meaning representation")
plt.xlabel("Number of training instances seen",size='xx-large')
plt.ylabel(r'\[ \displaysyle P(word~order) \]',size='xx-large')
#doAdj(probdir,figpdqa2)
doDet(probdir,figpdqa2)
doQuant(probdir,figpdqa2)


box = figpdqa2.get_position()
figpdqa2.set_position([box.x0, box.y0, box.width, box.height])
leg = figpdqa2.legend(loc='center right')


ppdqa.savefig()
ppdqa.close()



probdir = "newRepProbsN+1_bigSat_1W"
ppdqa = PdfPages(probdir+"/detqnadjplot_3reps.pdf")
figpdqafig = plt.figure(figsize=(rcParams['figure.figsize'][0],rcParams['figure.figsize'][1]))
figpdqa = figpdqafig.add_subplot(111)
plt.title("With referential uncertainty")

#doAdj(probdir,figpdqa)
#doDet(probdir,figpdqa)
#doQuant(probdir,figpdqa)

plt.xlabel("Number of training instances seen",size='xx-large')
plt.ylabel(r'\[ \displaysyle P(word~order) \]',size='xx-large')
box = figpdqa.get_position()
figpdqa.set_position([box.x0, box.y0, box.width, box.height])
leg = figpdqa.legend(loc='center right')

ppdqa.savefig()
ppdqa.close()
#figpdqafig.close()







probdir = "probOutputs/newRepProbsN+1_newlearnrate_MWE_"
pp = PdfPages(probdir+"/newRepProbsN+1_newlearnrate_MWE.pdf")
figsvo = plt.figure(figsize=(rcParams['figure.figsize'][0],rcParams['figure.figsize'][1]))
figtrans1 = figsvo.add_subplot(111)

doTrans(probdir+"_3reps",figtrans1,3,0)
#plt.title("Unambiguous meaning representation")
probdir = probdir
doTrans(probdir+"_5reps",figtrans1,5,6)

plt.xlabel("Number of training instances seen",size='xx-large')
plt.ylabel(r'\[ \displaysyle P(word~order) \]',size='xx-large')
leg = figtrans1.legend(loc='center right')#,bbox_to_anchor=(1,0.5))
#pp.savefig()
#pp.close()


#pp = PdfPages(probdir+"/svoruleplot_3reps.pdf")



#figsvo = plt.figure(figsize=(rcParams['figure.figsize'][0],rcParams['figure.figsize'][1]))
#figtrans2 = figsvo.add_subplot(111)
plt.xlabel("Number of training instances seen",size='xx-large')
plt.ylabel(r'\[ \displaysyle P(word~order) \]',size='xx-large')

#plt.title("With referential uncertainty")
#doTrans(probdir+"_3reps",figtrans1)

box = figtrans1.get_position()
#ax1.set_position([box.x0, box.y0, box.width, box.height])
leg = figtrans1.legend(loc='center right')#,bbox_to_anchor=(1,0.5))

pp.savefig()
pp.close()




#doIntrans(probdir)
