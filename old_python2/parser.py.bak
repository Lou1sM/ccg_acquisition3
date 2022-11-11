# this is a parser for a single sentence
from tools import generate_wordset
from tools import inf
from build_inside_outside_chart import *
from cat import synCat
from cat import cat
from inside_outside_calc import *
from sample_most_probable_parse import *
from exp import exp

# Will DEFINITELY need to prune this chart

def cky(chart1,sentence,minscores,RuleSet,beamsize):
    print "doing cky"
    verbose = False
    verbose = True
    chartsize = 0
    returnchart = {}
    for i in range(1,len(sentence)+1):
        returnchart[i] = {}
        #for c in chart1[0][1]:
        #	returnchart[1][c] = chart1[0][1][c]
        #	chartsize += 1
        for k in range(1,i+1):
            for c in chart1[i-k][i]:
                returnchart[k][c] = chart1[i-k][i][c]
                #print "adding ",c," to returnchart ",k
                chartsize += 1
    #print "\n\n\nlexical item returnchart is \n",returnchart
    for i in range(2,len(sentence)+1): ## length of span
        for j in range(len(sentence)-i+1): ## start of span
            for k in range(1,i): ## partition of span
                for lck in chart1[j][j+k]:
                    lc = chart1[j][j+k][lck]
                    for rck in chart1[j+k][j+i]:
                        #print "chartsize = ",chartsize
                        rc = chart1[j+k][j+i][rck]
                        lcat = lc.ccgCat
                        rcat = rc.ccgCat
                        #print "i is ",i," j is ",j," k is ",k
                        if verbose:
                            print "trying to combine ",lcat.toString()," with ",rcat.toString()
                            print "leftscore = ",lc.max_score
                            print "rightscore = ",rc.max_score

                            print "fwdapp"
                        newcat = lcat.copy().apply(rcat.copy(),"fwd")
                        if newcat:
                            if returnchart[i].has_key((newcat.synString(),newcat.semString(),j,j+i)):
                                ce = returnchart[i][(newcat.synString(),newcat.semString(),j,j+i)]
                            else:
                                ce = chart_entry(newcat,j,j+i,sentence)
                                # just according to inside score, but haven't been
                                # accumulating

                            target = lcat.syn.toString()+'#####'+rcat.syn.toString()
                            RuleSet.check_rule(ce.syn_key,lcat.syn,rcat.syn,"fwd",0)
                            rule_score = RuleSet.return_map_log_prob(ce.syn_key,target)
                            new_inside_score = rule_score+lc.inside_score+rc.inside_score
                            new_max_score = rule_score+lc.max_score+rc.max_score
                            ce.inside_score = log_sum(ce.inside_score,new_inside_score)
                            if new_max_score > ce.max_score:
                                ce.max_score = new_max_score
                            if verbose:
                                print "got newcat ",newcat.toString()," from ",lcat.toString()," ",rcat.toString()," maxscore is ",new_max_score
                                print "rule_score = ",rule_score

                            lc.add_parent(ce,'l')
                            rc.add_parent(ce,'r')
                            ce.add_child(((lc.syn_key,lc.sem_key,j,j+k),(rc.syn_key,rc.sem_key,j+k,j+i)))

                            chart1[j][j+i][(newcat.synString(),newcat.semString(),j,j+i)] = ce
                            returnchart[i][(newcat.synString(),newcat.semString(),j,j+i)] = ce
                            if len(chart1[j][j+i]) > beamsize:
                                removemin(chart1,j,j+i,minscores)
                            continue

                        if verbose: print "backapp"
                        newcat = rcat.copy().apply(lcat.copy(),"back")
                        if newcat:
                            if returnchart[i].has_key((newcat.synString(),newcat.semString(),j,j+i)):
                                ce = returnchart[i][(newcat.synString(),newcat.semString(),j,j+i)]
                            else:
                                ce = chart_entry(newcat,j,j+i,sentence)

                            target = lcat.syn.toString()+'#####'+rcat.syn.toString()
                            RuleSet.check_rule(ce.syn_key,lcat.syn,rcat.syn,"back",0)
                            rule_score = RuleSet.return_map_log_prob(ce.syn_key,target)
                            new_inside_score = rule_score+lc.inside_score+rc.inside_score
                            new_max_score = rule_score+lc.max_score+rc.max_score
                            ce.inside_score = log_sum(ce.inside_score,new_inside_score)
                            if new_max_score > ce.max_score:
                                ce.max_score = new_max_score

                            if verbose:
                                print "got newcat ",newcat.toString()," from ",lcat.toString()," ",rcat.toString()," maxscore is ",new_max_score
                                print "rule_score = ",rule_score

                            lc.add_parent(ce,'l')
                            rc.add_parent(ce,'r')
                            ce.add_child(((lc.syn_key,lc.sem_key,j,j+k),(rc.syn_key,rc.sem_key,j+k,j+i)))

                            chart1[j][j+i][(newcat.synString(),newcat.semString(),j,j+i)] = ce
                            returnchart[i][(newcat.synString(),newcat.semString(),j,j+i)] = ce
                            if len(chart1[j][j+i])  > beamsize:
                                removemin(chart1,j,j+i,minscores)

                            continue
                        if verbose: print "fwdcomp"
                        newcat = lcat.copy().compose(rcat.copy(),"fwd")
                        if newcat:
                            if returnchart[i].has_key((newcat.synString(),newcat.semString(),j,j+i)):
                                ce = returnchart[i][(newcat.synString(),newcat.semString(),j,j+i)]
                            else:
                                ce = chart_entry(newcat,j,j+i,sentence)


                            target = lcat.syn.toString()+'#####'+rcat.syn.toString()
                            RuleSet.check_rule(ce.syn_key,lcat.syn,rcat.syn,"fwd",1)

                            rule_score = RuleSet.return_map_log_prob(ce.syn_key,target)
                            new_inside_score = rule_score+lc.inside_score+rc.inside_score
                            new_max_score = rule_score+lc.max_score+rc.max_score
                            ce.inside_score = log_sum(ce.inside_score,new_inside_score)
                            if new_max_score > ce.max_score:
                                ce.max_score = new_max_score

                            if verbose:
                                print "got newcat ",newcat.toString()," from ",lcat.toString()," ",rcat.toString()," maxscore is ",new_max_score
                                print "rule_score = ",rule_score


                            lc.add_parent(ce,'l')
                            rc.add_parent(ce,'r')
                            ce.add_child(((lc.syn_key,lc.sem_key,j,j+k),(rc.syn_key,rc.sem_key,j+k,j+i)))

                            chart1[j][j+i][(newcat.synString(),newcat.semString(),j,j+i)] = ce
                            returnchart[i][(newcat.synString(),newcat.semString(),j,j+i)] = ce
                            chartsize += 1
                            if len(chart1[j][j+i])  > beamsize:
                                removemin(chart1,j,j+i,minscores)


                            continue
                        if verbose: print "backcomp"
                        newcat = rcat.copy().compose(lcat.copy(),"back")
                        if newcat:

                            if returnchart[i].has_key((newcat.synString(),newcat.semString(),j,j+i)):
                                ce = returnchart[i][(newcat.synString(),newcat.semString(),j,j+i)]
                            else:
                                ce = chart_entry(newcat,j,j+i,sentence)




                            target = lcat.syn.toString()+'#####'+rcat.syn.toString()
                            RuleSet.check_rule(ce.syn_key,lcat.syn,rcat.syn,"fwd",1)
                            rule_score = RuleSet.return_map_log_prob(ce.syn_key,target)
                            new_inside_score = rule_score+lc.inside_score+rc.inside_score
                            new_max_score = rule_score+lc.max_score+rc.max_score
                            ce.inside_score = log_sum(ce.inside_score,new_inside_score)
                            if new_max_score > ce.max_score:
                                ce.max_score = new_max_score


                            if verbose:
                                print "got newcat ",newcat.toString()," from ",lcat.toString()," ",rcat.toString()," maxscore is ",new_max_score
                                print "rule_score = ",rule_score


                            lc.add_parent(ce,'l')
                            rc.add_parent(ce,'r')
                            ce.add_child(((lc.syn_key,lc.sem_key,j,j+k),(rc.syn_key,rc.sem_key,j+k,j+i)))

                            chart1[j][j+i][(newcat.synString(),newcat.semString(),j,j+i)] = ce
                            returnchart[i][(newcat.synString(),newcat.semString(),j,j+i)] = ce
                            if len(chart1[j][j+i])  > beamsize:
                                removemin(chart1,j,j+i,minscores)


                            continue
                        if verbose: print "combination DID NOT WORK"
    if chart1[0][len(chart1)]!={}:
        print "PARSED:: ",sentence,chart1[0][len(chart1)]
    if verbose:
        print "\n\ninside scores "
        for level in returnchart:
            for item in returnchart[level]:
                print item
                entry = returnchart[level][item]
                print entry.toString(),"  ",entry.inside_score
        print "\n\n"

    #print "\n\n\n",returnchart
    return returnchart

def removemin(chart1,start,end,minscores):
    minscore = inf
    secondmin = inf
    minc = None
    for c in chart1[start][end]:
        ce = chart1[start][end][c]
        if ce.inside_score <= minscore:
            minc = c
            secondmin = minscore
            minscore = ce.inside_score

    del chart1[start][end][minc]
    minscores[start][end]=secondmin

def get_parse_chart(sentence,sem_store,RuleSet,lexicon,sentence_count,test_out):
    guesslex = True # False
    guesslex = True
    beamsize = 100/(len(sentence)+1)
    #beamsize = 1
    print "beam is ",beamsize
    verbose = False # True
    verbose = True
    print "PARSING: really should prune lexicon (and rules??)",sentence
    # this parser should build up the chart in a far
    # more efficient way than the previous one.
    # we almost certainly can integrate the inside/outside chart
    # Build one chart up from bottom
    # Walk back down and sample
    wordset = generate_wordset(sentence,len(sentence))
    chart1 = {}
    minscores = {}
    for i in range(0,len(sentence)):
        sc = {}
        ms = {}
        for j in range(i+1,len(sentence)+1):
            sc[j]={}
            ms[j]=[]
        chart1[i] = sc
        minscores[i] = ms
    if verbose: print "wordset is ",wordset
    for level in wordset:
        if verbose: print "level is ",level
        for word in wordset[level]:
            # this is all kinda bollox
            # make a better span set (i,j,cat etc)
            start = 0
            for w in word:
                if w.count(" ")!=len(sentence)-level:
                    #print "level is ",level," len sentence is ",len(sentence)," rejecting ",w," with spacecount ",w.count(" ")
                    continue
                end = start+w.count(" ")+1
                #print 'word is ',w," space count is ",w.count(" ")
                if verbose: print "start is ",start," end is ",end," for ",w
                if end-start>1 and lexicon.mwe == False: continue
                poss_lex = []
                poss_lex = lexicon.get_lex_items(w,test_out,guesslex,sem_store,beamsize)
                if verbose: print w,' has ',len(poss_lex),' realisations'
                if len(poss_lex) > 0:
                    #print 'most probable is ',poss_lex[0]
                    for pl in poss_lex:
                        if verbose: print "pl is ",pl
                        l = pl
                        #print "l is ",l
                        if verbose: print l.toString()
                        #print "here"
                        syncat = synCat.readCat(l.syn)
                        if verbose: print "syn is ",l.syn," syncat is ",syncat.toString()
                        sem = sem_store.get(l.sem_key) # exp.makeExpWithArgs(l.sem_key,{})[0]
                        if verbose: print "sem is ",sem.toString(True)
                        c = cat(syncat,sem)
                        ce = chart_entry(c,start,end,sentence)
                        if hasattr(l, 'is_shell_item') and l.is_shell_item:
                            # if it's a shell, there is the probability of it generating a new LF and the probability
                            # of a new LF
                            ce.word_score = \
                                lexicon.get_map_word_given_shell(ce.word_target,ce.syn_key,ce.sem_key,sentence_count,sem_store)
                            # all the probability goes into word_score
                            ce.sem_score = 0.0
                        else:
                            ce.word_score = \
                                lexicon.get_map_log_word_prob(ce.word_target,ce.syn_key,ce.sem_key,sentence_count)
                            ce.sem_score = lexicon.get_map_log_sem_prob(ce.syn_key,ce.sem_key,sem_store)

                        print "sem score for ",ce.syn_key," -> ",ce.sem_key," = ",ce.sem_score
                        if verbose: print ce.syn_key+" -> "+ce.sem_key+" = "+str(ce.sem_score)
                        rule_score = RuleSet.return_map_log_prob(ce.syn_key,ce.syn_key+'_LEX')
                        if verbose: print ce.syn_key+"_LEX"+" = "+str(rule_score)
                        ce.inside_score = ce.word_score+ce.sem_score+rule_score
                        ce.max_score = ce.word_score+ce.sem_score+rule_score
                        if len(chart1[start][end])<beamsize or minscores[start][end]<ce.inside_score:
                            chart1[start][end][(l.syn,l.sem_key,start,end)] = ce
                            if verbose: print "added ",ce.toString()," to ",start,end
                            if len(chart1[start][end]) > beamsize:
                                removemin(chart1,start,end,minscores)
                        elif verbose: print "not adding ",ce.toString()
                start += w.count(" ")+1
    returnchart = cky(chart1,sentence,minscores,RuleSet,beamsize)
    return returnchart


def parse(sentence,sem_store,RuleSet,lexicon,sentence_count,test_out,test_out_parses=None,target_top_cat=None):
    verbose = False
    returnchart = get_parse_chart(sentence,sem_store,RuleSet,lexicon,sentence_count,test_out)
    if returnchart is None:
        return (None,None,None)
    if returnchart is not None:
        if verbose:
            print "\n\n\n\n\nDOING NUMBERS FOR CHART :::",returnchart,"\n\n\n\n "
                #i_o_oneChart(chart1,sem_store,lexicon,RuleSet,False,0.0,0)
        topparses = []
        for entry in returnchart[len(returnchart)]:
            top = returnchart[len(returnchart)][entry]
            topcat = top.syn_key
            if target_top_cat and topcat != target_top_cat:
                continue
            top.inside_score = top.inside_score
            usestartrule = True
            if usestartrule:
                RuleSet.check_start_rule(top.ccgCat.syn)
                top.outside_score  = RuleSet.return_map_log_prob("START",topcat)
            else:
                top.outside_score = 0.0

            if verbose: print "top is ",top
            topparses.append((top.inside_score+top.outside_score,top))

        if len(topparses)==0:
            return (None,None,None)
        topparses.sort()
        topparses.reverse()
        if verbose:
            print "\n\n"
            for t in topparses:
                print "parse : ",t[0],sample(t[1],returnchart,RuleSet)," maxscore = ",t[1].max_score
            print "\n\n"
        if test_out_parses:
            print >> test_out_parses, "\n\n"
            for t in topparses:
                print >> test_out_parses, "parse : ",t[0],sample(t[1],returnchart,RuleSet),\
                    " maxscore = ",t[1].max_score
            print >> test_out_parses, "\n\n"

        topnode = sorted(topparses)[-1][1]
        top_parse = sample(topnode,returnchart,RuleSet)
        print 'top parse:'
        print  top_parse
        print  topnode.inside_score
        print  "\n"

        if test_out_parses:
            print >> test_out_parses, 'top parse:'
            print >> test_out_parses, top_parse
            print >> test_out_parses, topnode.inside_score
            print >> test_out_parses, "\n"
        return (topnode.ccgCat.sem,top_parse,topnode.ccgCat)



