# this is to generate a sentence from a logical expression.
from build_inside_outside_chart import *
from inside_outside_calc import i_o_oneChart
from sample_most_probable_parse import sample
#from sample_most_probable_parse import get_parse_prob
from math import exp
from errorFunct import error

def assignWords(chart,lexicon):
    for level in chart:
        print "level is ",level
        toadd = []
        for item in chart[level]:
            entry = chart[level][item]
            ccgcat = entry.ccgCat
            w_syn_sem = lexicon.getMaxWordForSynSem(ccgcat.syn.toString(),ccgcat.sem.toString(True))
            if w_syn_sem is None: continue
            entry.words = w_syn_sem[0]
            entry.word_target =  w_syn_sem[0]
    
# can't just sample down since we need to generate the correct semantics
def generateSent(lexicon,RuleSet,topCat,catStore,sem_store,oneWord,corrSent,genoutfile,sentence_count,sentnum):
    # pack 'word list' with None so that we can reuse old code
    # generate words at leaves
    # do MAP inside-outside
    wordlist = ["placeholderW"]*(len(topCat.sem.allSubExps()))
    chart = build_chart([topCat],wordlist,RuleSet,lexicon,catStore,sem_store,oneWord)
    assignWords(chart,lexicon)
    i_o_oneChart(chart,sem_store,lexicon,RuleSet,False,0.0,lexicon.sentence_count,True)    
    topparses = []
    if len(chart[len(chart)])!=1: error()
    for entry in chart[len(chart)]:
        top = chart[len(chart)][entry]
        topparses.append((top.inside_score,top))
    
    top_parse = sample(sorted(topparses)[-1][1],chart,RuleSet)
    print >> genoutfile, "\ntop generated parse"+str(sentnum)+":"
    print  >> genoutfile, top_parse
    print  >> genoutfile, top.inside_score
    
    print "\ntop generated parse"+str(sentnum)+":"
    print top_parse
    print top.inside_score
    
    chart = build_chart([topCat],corrSent.split(),RuleSet,lexicon,catStore,sem_store,oneWord)
    if chart is not None:
        corr_score = i_o_oneChart(chart,sem_store,lexicon,RuleSet,False,0.0,lexicon.sentence_count)
        print >> genoutfile, "corr score"+str(sentnum)+" is ",corr_score
        print  >> genoutfile,"prob gen corr"+str(sentnum)+" = ",exp(corr_score - top.inside_score)
        print  "prob gen corr"+str(sentnum)+" = ",exp(corr_score - top.inside_score)
        
    

