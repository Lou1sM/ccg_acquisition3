from matplotlib import rcParams
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def getprobs(word,targetsem,probfile,outfile=None):
    problist = [0]
    instlist = [0]
    for i in range(len(probfile)):
        line = probfile[i]
        if line.find(word+"  ->  "+targetsem)!=-1:
            print >> outfile, line,
            prob = float(line.split("::")[0].split()[-1].strip().rstrip())
            inst = int(line.split("::")[1].strip().rstrip())
            problist.append(prob)
            instlist.append(inst)
    return (problist,instlist)

probfile = open(raw_input("where is the wordprob file?\n"),"r").readlines()
markerlist = ['r','b--','g--','rs','bs','gs','r^','b^','g^']
def doQn():
    pp = PdfPages("quantprobs/quantplot.pdf")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    quants = ['some','another','an','any','all','both']

    dets = ['the','a']
    leglist = []
    i = 0
    for q in quants:
        print "doing :: ",q
        outfile = open("quantprobs/"+q,"w")
        targetsem = "lambda $0_{<e,t>}.qn|"+q+"($1,$0($1))"
        latextarget = r'\[ \displaystyle {\rm '+q+r'}  \rightarrow \lambda f . '+q+r'(x,f(x)) \]'
        (instlist,problist) = getprobs(q,targetsem,probfile,outfile)
        if problist[-1]>0:
            ax1.plot(problist,instlist,label=latextarget)
        leglist.append(q)
        i+=1
    for d in dets:
        print "doing :: ",d
        outfile = open("quantprobs/"+d,"w")
        targetsem = "lambda $0_{<e,t>}.det|"+d+"($1,$0($1))"
        latextarget = r'\[ \displaystyle {\rm '+d+r'} \rightarrow \lambda f. '+d+r'(x,f(x))\]'

        (instlist,problist) = getprobs(d,targetsem,probfile,outfile)
        ax1.plot(problist,instlist,label=latextarget)
        leglist.append(d)
        i+=1
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    leg = ax1.legend(loc='center left',bbox_to_anchor=(1,0.5))
    #show()
    pp.savefig()
    pp.close()

############ NOUNS #############
def doNoun():
    pp = PdfPages("nounprobs/nounplot.pdf")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    nouns = ['spoon','cup','pencil','name','coffee','lady','drink','duck']
    leglist = []
    i = 0
    for n in nouns:
        print "doing :: ",n
        outfile = open("nounprobs/"+n,"w")
        targetsem = "lambda $0_{e}.n|"+n+"($0)"
        latextarget = r'\[\displaystyle {\rm '+n+r'}\rightarrow \lambda x'+n+r'(x)\]'
        (instlist,problist) = getprobs(n,targetsem,probfile,outfile)
        ax1.plot(problist,instlist,label=latextarget)
        leglist.append(n)
        i+=1
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    leg = ax1.legend(loc='center left',bbox_to_anchor=(1,0.5))
    #show()

    pp.savefig()
    pp.close()

############ Ditransitive Verbs #############
def doDitrans():
    pp = PdfPages("ditransprobs/ditransplot.pdf")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    verbs = ['give','bring','find','write','tell']
    leglist = []
    i = 0
    for v in verbs:
        print "doing :: ",v
        outfile = open("ditransprobs/"+v,"w")
        targetsem = "lambda $0_{e}.lambda $1_{e}.lambda $2_{e}.lambda $3_{ev}.v|"+v+"($1,$0,$2,$3)"
        latextarget = r'\[ \displaystyle {\rm '+v.replace("&","")+r'} \rightarrow \lambda x \lambda y \lambda z \lambda ev .'+v.replace("&","")+r'(y,x,z,ev )\]'
        (instlist,problist) = getprobs(v,targetsem,probfile,outfile)
        ax1.plot(problist,instlist,label=latextarget)
        leglist.append(v)
        i+=1
        targetsem = "lambda $0_{e}.lambda $1_{e}.lambda $2_{e}.lambda $3_{ev}.v|"+v+"($0,$2,$1,$3)"
        latextarget = r'\[ \displaystyle {\rm '+v.replace("&","")+r'} \rightarrow\lambda x \lambda y \lambda z \lambda ev .'+v.replace("&","")+r'"(x,z,y,ev ) \]'
        (instlist,problist) = getprobs(v,targetsem,probfile,outfile)
        ax1.plot(problist,instlist,label=latextarget)
        leglist.append(v)
        i+=1
        targetsem = "lambda $0_{e}.lambda $1_{e}.lambda $2_{e}.lambda $3_{ev}.v|"+v+"($0,$1,$2,$3)"
        latextarget = r'\[ \displaystyle {\rm '+v.replace("&","")+r'} \rightarrow\lambda x \lambda y \lambda z \lambda ev .'+v.replace("&","")+r'(x,y,z,ev) \]'
        (instlist,problist) = getprobs(v,targetsem,probfile,outfile)
        ax1.plot(problist,instlist,label=latextarget)
        leglist.append(v)
        i+=1
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    leg = ax1.legend(loc='center left',bbox_to_anchor=(1,0.5))
    #show()

    pp.savefig()
    pp.close()


#['dancing','sit','went','blow']
#sit  ->  lambda $0_{e}.lambda $1_{ev}.v|sit($0,$1)  =  0.815386856882  ::  980
def doIntrans():
    pp = PdfPages("intransprobs/intransplot.pdf")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    verbs = ['dancing','sit','went','blow']
    leglist = []
    i = 0
    for v in verbs:
        print "doing :: ",v
        outfile = open("ditransprobs/"+v,"w")
        targetsem = "lambda $0_{e}.lambda $1_{ev}.v|"+v+"($0,$1)"
        latextarget = r'\[ \displaystyle {\rm '+v.replace("&","")+r'}\rightarrow \lambda x \lambda ev .'+v.replace("&","")+r'(x,ev ) \]'

        (instlist,problist) = getprobs(v,targetsem,probfile,outfile)
        ax1.plot(problist,instlist,label=latextarget)
        leglist.append(v)
        i+=1
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    leg = ax1.legend(loc='center left',bbox_to_anchor=(1,0.5))
    #show()

    pp.savefig()
    pp.close()



#['doing','put','have','want','like','say','take','dropped','open','peel','pull','fall']
#doing  ->  lambda $0_{e}.lambda $1_{e}.lambda $2_{ev}.part|do-PROG($0,$1,$2)
#doing  ->  lambda $0_{e}.lambda $1_{e}.lambda $2_{ev}.part|do-PROG($1,$0,$2)
def doTrans():
    pp = PdfPages("transprobs/transplot.pdf")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    verbs = ['doing','put','have','want','like','say','take','dropped','open','peel','pull','fall']
    verbswithsem = [('doing', 'part|do-PROG'), ('put', 'v|put&ZERO'), ('have', 'v|have'), ('want', 'v|want'),\
                        ('like', 'v|like'), ('say', 'v|say'), ('take', 'v|take'), ('dropped', 'v|drop-PAST'),\
                        ('open', 'v|open'), ('peel', 'v|peel'), ('pull', 'v|pull'), ('fall','v|fall')]
    leglist = []
    i = 0
    for (v,s) in verbswithsem:
        print "doing :: ",v
        outfile = open("transprobs/"+v,"w")
        targetsem = "lambda $0_{e}.lambda $1_{e}.lambda $2_{ev}."+s+"($0,$1,$2)"
        latextarget = r'\[ \displaystyle {\rm '+v.replace("&","")+r'}\rightarrow \lambda x \lambda y \lambda ev . '+s+r'(x,y,ev ) \]'
        (instlist,problist) = getprobs(v,targetsem,probfile,outfile)
        ax1.plot(problist,instlist,label=latextarget)
        leglist.append(v)
        i+=1

        targetsem = "lambda $0_{e}.lambda $1_{e}.lambda $2_{ev}."+s+"($1,$0,$2)"
        latextarget = r'\[ \displaystyle {\rm '+v.replace("&","")+r'} \rightarrow \lambda x \lambda y \lambda ev . '+s+r'(y,x,ev) \]'
        (instlist,problist) = getprobs(v,targetsem,probfile,outfile)
        ax1.plot(problist,instlist,label=latextarget)
        leglist.append(v)
        i+=1
    #show()

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    leg = ax1.legend(loc='center left',bbox_to_anchor=(1,0.5))

    pp.savefig()
    pp.close()



#['now','better','just','later','again','already','upstairs','first','almost']
#now  ->  lambda $0_{<ev,t>}.lambda $1_{ev}.and($0($1),adv|now($1))
def doAdv():
    pp = PdfPages("advprobs/advplot.pdf")
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    advs = ['now','better','just','later','again','already','upstairs','first','almost']
    leglist = []
    i = 0
    for a in advs:
        print "doing :: ",a
        outfile = open("advprobs/"+a,"w")

        targetsem = "lambda $0_{<ev,t>}.lambda $1_{ev}.and($0($1),adv|"+a+"($1))"

        latextarget = r'\[ \displaystyle {\rm '+a+r' } \rightarrow \lambda f \lambda ev. '+a+r'(ev) \wedge f(ev)\]'
        (instlist,problist) = getprobs(a,targetsem,probfile,outfile)
        ax1.plot(problist,instlist,label=latextarget)
        leglist.append(a)
        i+=1

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    leg = ax1.legend(loc='center left',bbox_to_anchor=(1,0.5))
    #show()

    pp.savefig()
    pp.close()



##### run the things #######
doQn()
#doDitrans()
#doNoun()
#doIntrans()
#doTrans()
#doAdv()
