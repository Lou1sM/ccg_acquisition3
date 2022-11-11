# inside-outside maths
from math import log
from math import exp
from tools import log_sum
from tools import log_diff
from tools import inf
import lexicon_classes

def i_o(sentence_charts,sem_store,lexicon,RuleSet,old_log_prob):

    log_prob = 0
    # Get inside probs
    for c in sentence_charts:
        #print 'sc is ', sentence_charts[c]
        #print '\n\n'
        pass
# really need to work out what this learning rate means.... 
# delta alpha = (1/gamma) * (eta(tau) * T * E(param) + gamma0*alpha0 - gamma*alpha(tau-1))
# eta(tau) = (1+ lambda(tau)/eta(n-1))^-1
#discountfact = 

def i_o_oneChart(chart,sem_store,lexicon,RuleSet,doupdates,old_norm,\
                     sentence_count,generating=False,sentToGen=None):
    #    lexicon.set_temp_params()
    #    RuleSet.set_temp_params()
    
    #k = 0.0001
    #To = 10
    #df = 0.0
    # print "sentence count is ",sentence_count
    #for i in range(1,sentence_count+1):
    #    prod = 1.0
    #    for j in range(i+1,sentence_count+1):
    #        prod = prod*(1.0 - 1.0/((j-2)*k+To))
            #print "df now ",df
    #    df += prod
    #if sentence_count+1<3:
    #    df = 1.0 - 1.0/To
    
    #if sentence_count%10==0:
    #    lexicon.printSynSem()
    datasize = 4000
    gamma = 1.0/datasize
    #    learningrate = 1.0/df

    # won't actually know the data size 

    

    
    #print "df is ",df
    #print "LR is ",learningrate

    RulesUsed = []
    LexItemsUsed = []
    # sem:syn:word
    wordFromSemSyn = {}
    semFromSyn = {}
    verbose = True
    verbose = False
    
    #print "\n\n"
    # bottom up, get inside and sum of number of parses
    for level in chart:
        for item in chart[level]:
            entry = chart[level][item]
            #entry.word_prob = lexicon.get_word_prob(entry.word_target,entry.syn_key,entry.sem_key)
            entry.word_score = lexicon.get_log_word_prob(entry.word_target,entry.syn_key,entry.sem_key,sentence_count)
            
            if generating and entry.word_score != -inf:
                print "non zero prob for ",entry.word_target,entry.syn_key,entry.sem_key
                print "word prob is ",entry.word_score
            #if
            #print entry.syn_key,entry.sem_key," -> ",entry.word_target," = ",entry.word_score
            #entry.sem_prob = lexicon.get_sem_prob(entry.syn_key,entry.sem_key,sem_store)
            entry.sem_score = lexicon.get_log_sem_prob(entry.syn_key,entry.sem_key,sem_store)
            if verbose: 
                print entry.syn_key," -> ",entry.sem_key," = ",entry.sem_score

            if not entry.word_score == -inf: 
                if not wordFromSemSyn.has_key(entry.sem_key+":"+entry.syn_key+":"+entry.word_target):
                    wordFromSemSyn[entry.sem_key+":"+entry.syn_key+":"+entry.word_target] = entry.word_score
                if not semFromSyn.has_key(entry.sem_key+":"+entry.syn_key):
                    semFromSyn[entry.sem_key+":"+entry.syn_key] = entry.sem_score
                
            # should probably have a distinct node for the semantic
            # part of this
            
            #rule_prob = RuleSet.return_prob(entry.syn_key,entry.syn_key+'_LEX')
            rule_score = RuleSet.return_log_prob(entry.syn_key,entry.syn_key+'_LEX')
            if verbose:
                print entry.syn_key+'_LEX'," = ",rule_score
                print "lex alpha = ",RuleSet.Rules[RuleSet.Targets[entry.syn_key+'_LEX']].Targets[entry.syn_key+'_LEX'].alpha
                print "tot alpha = ",RuleSet.Rules[RuleSet.Targets[entry.syn_key+'_LEX']].alpha_tot
            # Need some new error checking in here 
            #if rule_prob == 0:
                #print 'zero prob for ',entry.syn_key+'_LEX'
            #print 'rule_prob is ',rule_prob,' for ',entry.cat.key+'_LEX'
            #if entry.word_prob == 0:
                #print 'word prob is zero for ',entry.word_target
            #if entry.sem_prob == 0:
                #print 'sem prob is zero for ',entry.sem_key
            
            entry.inside_score = entry.word_score+entry.sem_score+rule_score
            #entry.inside_prob = entry.word_prob*entry.sem_prob*rule_prob
            
            entry.addNumParses(1)
            
            rule_score = -inf
            for pair in entry.children:
                left_syn = pair[0][0]
                right_syn = pair[1][0]
                target = left_syn+'#####'+right_syn
                #rule_prob = RuleSet.return_prob(entry.syn_key,target)
                rule_score = RuleSet.return_log_prob(entry.syn_key,target)

                if verbose:
                    print target," = ",rule_score

                entryL = chart[pair[0][3]-pair[0][2]][pair[0]]
                entryR = chart[pair[1][3]-pair[1][2]][pair[1]]
                #new_inside_prob = rule_prob*entryL.inside_prob*entryR.inside_prob
                new_inside_score = rule_score+entryL.inside_score+entryR.inside_score
                #entry.inside_prob = entry.inside_prob + new_inside_prob
                entry.inside_score = log_sum(entry.inside_score,new_inside_score)
                entry.addNumParses(entryL.getNumParses()*entryR.getNumParses())
                
                
#                 print 'inside prob for ',entry.sem_key,entry.cat.key,entry.words,' is ',new_inside_prob
#                 print 'product of ',entryL.sem_key,entryL.cat.key,entryL.words,' and ',entryR.sem_key,entryR.cat.key,entryR.words
#                 print 'probs of these are ',entryL.inside_prob,entryR.inside_prob
#                 print 'rule_prob is ',rule_prob,' for ',target

                #if RuleSet.return_prob(entry.syn_key,target) == 0:
                    #print 'zero prob for ',target


            #if entry.inside_prob > 1.0:
                #print 'inside prob is over one'
                #print item
                #print 'rule prob is ',rule_prob
            #if entry.inside_prob == 0.0:
                #print 'inside prob is zero for ',entry.word_target,entry.syn_key,entry.sem_key
    # Get outside probs #

    if verbose:
        print "\n\ninside scores "
        for level in chart:
            for item in chart[level]:
                entry = chart[level][item]
                print entry.toString(),"  ",entry.inside_score
        print "\n\n"
     
    top_down = range(1,len(chart)+1)
    top_down.reverse()
    #for level in top_down:
        #for item in chart[level]:
            #entry = chart[level][item]
            #for pair in entry.children:
                #foundparentl = False
                #foundparentr = False
                
                #entryL = chart[pair[0][3]-pair[0][2]][pair[0]]
                #entryR = chart[pair[1][3]-pair[1][2]][pair[1]]
                #for parent in entryL.parents: 
                    #if parent[0]==entry and parent[2]=="l": foundparentl = True
                #for parent in entryR.parents: 
                    #if parent[0]==entry and parent[2]=="r": foundparentr = True
                #if not (foundparentl and foundparentr):
                    #print "l ",foundparentl
                    #print "r ",foundparentr
                    #print entry.toString()
                    #print entryL.toString()
                    #print entryR.toString()
                    
                    #error()
            
   
    for level in top_down:
        #print "level top down is ",level
        for item in chart[level]:
            entry = chart[level][item]
            #print "entry is ",entry.toString()
            usestartrule = True
            if len(entry.parents) == 0:
                topcat = chart[len(chart)][item].syn_key
                if usestartrule: 
                    RuleSet.check_start_rule(chart[len(chart)][item].ccgCat.syn)
                    entry.outside_score = RuleSet.return_log_prob("START",topcat)        
                else: entry.outside_score = 0.0
                entry.outside_prob = exp(entry.outside_score)

            else:
                for parent in entry.parents:
                    # do we get all the parents??
                    father = parent[0]

                    pair = father.children[parent[1]]
                    side = parent[2]
                    
                    #p_out_prob = father.outside_prob
                    p_out_score = father.outside_score
                    #if p_out_prob == 0:
                        #print 'zero error 1'
                        
                    left_syn = pair[0][0]
                    right_syn = pair[1][0]
                    target = left_syn+'#####'+right_syn    
                    #rule_prob = RuleSet.return_prob(father.syn_key,target)        
                    rule_score = RuleSet.return_log_prob(father.syn_key,target)        
                    #if rule_prob == 0:
                        #print 'zero error 2'
                    s_in_prob = 0
                    if side == 'l':
                        s_in_score = chart[pair[1][3]-pair[1][2]][pair[1]].inside_score    
                    elif side == 'r':
                        s_in_score = chart[pair[0][3]-pair[0][2]][pair[0]].inside_score
                    #if s_in_prob == 0:
                        #print 'zero error 3 for , ',entry.sem_key
                        #print 'side is ',side
                        #print 'sem is ',chart[pair[0][3]-pair[0][2]][pair[0]].word_target
                        #print 'words are ',chart[pair[0][3]-pair[0][2]][pair[0]].sem_key
                    #entry.outside_prob += p_out_prob*rule_prob*s_in_prob
                    entry.outside_score = log_sum(entry.outside_score,p_out_score+s_in_score+rule_score)
    
    norm_score = -inf
    # should go up to where inside is done
    # Really do need to put a START node in here
    for item in chart[top_down[0]]:
        topcat = chart[top_down[0]][item].syn_key
        norm_score = log_sum(norm_score,chart[top_down[0]][item].inside_score+chart[top_down[0]][item].outside_score)

    if norm_score == -inf : return
    for item in chart[top_down[0]]:
        topcat = chart[top_down[0]][item].syn_key
        log_score_start = chart[top_down[0]][item].inside_score+chart[top_down[0]][item].outside_score - norm_score
        RuleSet.store_log_update(topcat,log_score_start)


    lexItemUpdates = {}
    print "norm score is ",norm_score
    #######################################
    # This is the probability update bit  #
    #######################################
    # DON'T LOCALLY NORMALISE 
    #######################################
    
    #onewordprobs = {}
    #for i in range(len(chart)): onewordprobs[i]=-inf
    #for level in chart:
        #for item in chart[level]:
            #entry = chart[level][item]
            #if len(entry.words)==1: 
                ##print "entry with scores is ",entry.toString()," inside is ",entry.inside_score," outside is ",entry.outside_score
                #rule_score = RuleSet.return_log_prob(entry.syn_key,entry.syn_key+'_LEX')
                #onewordprobs[entry.p] = log_sum(onewordprobs[i],entry.inside_score+entry.outside_score)
    #for i in onewordprobs:
        ##print "onewordprobs for ",i," is ",onewordprobs[i]," norm is ",norm_score
    
    onewordupdates = {}
    for i in range(len(chart)): onewordupdates[i]=-inf
    for level in chart:
        #print "level is ",level
        for item in chart[level]:
            #print "item is ",item
            entry = chart[level][item]
            #print "entry is ",entry.toString()
            #B_pq = entry.inside_prob
            B_pq = entry.inside_score
            #a_pq = entry.outside_prob
            a_pq = entry.outside_score

            node_score = B_pq+a_pq
            
            #if len(entry.words)==1: print "p is ",entry.p," node score is ",node_score," norm score is ",norm_score
            
            node_sum = -inf
            node_inside_sum = -inf
            for pair in entry.children:
                l_child = chart[pair[0][3]-pair[0][2]][pair[0]]
                r_child = chart[pair[1][3]-pair[1][2]][pair[1]]
                left_syn = l_child.syn_key
                right_syn = r_child.syn_key
                
                target = left_syn+'#####'+right_syn
                # want the rule score
                rule_score = RuleSet.return_log_prob(entry.syn_key,target)
                # should the outside go here???
                child_score = l_child.inside_score + r_child.inside_score
                                
                # E(rule) += (child_inside * parent_outside * rule_prob)/norm
                
                logRuleExp = (child_score + a_pq + rule_score) - norm_score
                
                #node_sum = log_sum(node_sum,child_score)
                # we have the outside score, which tells us the prob of
                # the parent node being true. there is only one way of 
                # getting from the child pair to the parent: through this
                # rule. So this is the ruleExp.
                #logRul eExp = child_score - norm_score
                # not sure this is right
                node_inside_sum = log_sum(node_inside_sum,child_score + rule_score)
                
                if logRuleExp >= 0.0:
                    print "logRuleExp = ",logRuleExp
                    print "child score is ",child_score
                    print "outside is ",a_pq
                    print "rule score is ",rule_score
                    print "norm score is ",norm_score
                #print "logER is ",logER
                RuleSet.store_log_update(target,logRuleExp)
                
                #l = chart[pair[0][3]-pair[0][2]][pair[0]]
                #r = chart[pair[1][3]-pair[1][2]][pair[1]]
                ##B_pd = l.inside_prob
                ##B_dq = r.inside_prob
                #B_pd = l.inside_score
                #B_dq = r.inside_score
                
                #left_syn = pair[0][0]
                #right_syn = pair[1][0]
                #target = left_syn+'#####'+right_syn
                ###################################                    
                ## not sure what is going on here #
                ## or indeed why                  #
                ###################################                    
                ##RuleSet.update_target_p(a_pq,B_pd,B_dq,target)
                
                #rule_prob = 
                #RuleSet.update_target_p(a_pq,B_pd,B_dq,target)
                
                #if not target in RulesUsed:
                    #RulesUsed.append(target)
                ##################################
                ##################################
            #print "node sum is ",node_sum
            #print "norm score is ",norm_score
            #print "node score is ",node_score
            if node_score==-inf:
                #print "inf node score for ",entry.toString()
                if len(entry.words)==1: print "inf for one word"
                continue

            target = entry.syn_key+'_LEX'    
            rule_score = RuleSet.return_log_prob(entry.syn_key,target)

            # really need to work out what this should be, but e^1e-5 is small
            # think about norm though

            if 1E-5>=entry.inside_score-node_inside_sum>=-1E-5:
            #if B_pq - node_inside_sum
                if entry.q-entry.p==1: print "lexscore is -inf for ",entry.lexKey()
                lex_score = -inf
            else:
                #print node_score-norm_score,">",node_sum
                #lex_score = log_diff(node_score,node_sum)
                lex_score = log_diff(entry.inside_score,node_inside_sum)
                #print "lex score is ",lex_score," for ",entry.lexKey()
                #print "node score is ",node_score
           
            logLexExp = (lex_score + a_pq)  - norm_score

            if len(entry.words)==1: 
                if verbose:
                    if node_inside_sum!=-inf: print "for ",entry.word_target," insidesum is ",node_inside_sum
                #print "lexDat for ",entry.toString()," lex = ",lex_score," outside = ",a_pq," rule score = ",\
                #rule_score," norm score = ",norm_score
                onewordupdates[entry.p]=log_sum(onewordupdates[entry.p],logLexExp)
                            
            #logLexExp = logLexExp  - norm_score
            #if lex_
            #print "lex score for ",entry.toString()," is ",lex_score
            #target = entry.syn_key+'_LEX'
            #lex_prob = RuleSet.return_prob(entry.syn_key,target)
            #lex_score = RuleSet.return_log_prob(entry.syn_key,target)
            #word_prob = lexicon.get_word_prob(entry.word_target,entry.syn_key,entry.sem_key)
            #word_score = lexicon.get_log_word_prob(entry.word_target,entry.syn_key,entry.sem_key)
            #logER = lex_score + word_score
            #RuleSet.update_target_log_p(a_pq,1.0,word_prob,target)
            #RuleSet.update_target_log_p(logER,target)
            if logLexExp >= 0.0:
                print "cell is ",entry.toString()
                print "entry inside is ",entry.inside_score
                print "inside sum is ",node_inside_sum
                
                print "lex score  = ",lex_score
                
                #print "child score is ",child_score
                print "outside is ",a_pq
                #print "rule score is ",rule_score
                print "norm score is ",norm_score
                #lex_score = log_diff(entry.inside_score,node_inside_sum)
            
            RuleSet.store_log_update(target,logLexExp)
            # this only needs to be done once for each time                                                            
            # the rule head is seen
            #RuleSet.update_bottom(entry.syn_key,a_pq,B_pq)
            # 
            lexicon.store_log_update(entry.word_target,entry.syn_key,entry.sem_key,logLexExp)
            #            if verbose:
            #            print "update for ",entry.lexKey()," is ",logLexExp
            #print lex_score
            #print norm_score
            if lex_score != -inf:
                if len(entry.words)>1 and lexicon_classes.syn_sem.one_word: 
                    print "\nERROR, NONZERO UPDATE FOR MWE ",entry.lexKey()
                    print "node inside ",entry.inside_score
                    print "node score ",node_score
                    print "norm score ",norm_score
                    print "node inside sum ",node_inside_sum
                    print "diff is ",entry.inside_score-node_inside_sum
                    print "lex score ",lex_score,"\n"
                    print "lex alpha ",lexicon.lex[(entry.word_target,entry.syn_key,entry.sem_key)].alpha
                if not lexItemUpdates.has_key(entry.lexKey()): 
                    lexItemUpdates[entry.lexKey()] = logLexExp
                else:  lexItemUpdates[entry.lexKey()] += logLexExp
            elif len(entry.words)==1:
                print "\nlex score is -inf for ",entry.lexKey()
                #print "node score ",node_score
                #print "norm score ",norm_score
                #print "node inside sum ",node_inside_sum
                #print "lex score ",lex_score,"\n"
                
                
        
            #entry.sem_prob = lexicon.get_sem_prob(entry.syn_key,entry.sem_key,sem_store)        
            #if not (entry.word_target,entry.cat.key,entry.sem_key) in LexItemsUsed:
            #LexItemsUsed.append((entry.word_target,entry.cat.key,entry.sem_key))
    
    #for i in range(len(chart)): print "update sum for ",i," is ",onewordupdates[i]
    #if not entry.word_score == -inf: 
    #print "\n\nLexProbs:"
    #print "wordProbs"
    #sortedl = []
    #for probentry in sorted(wordFromSemSyn.keys()): 
        #print probentry," :: ", wordFromSemSyn[probentry]
    #print "\nsemProbs"
    #for probentry in sorted(semFromSyn.keys()):
        #print probentry," :: ", semFromSyn[probentry]
        
        
        
    wordstokeys = {} 
    for probupdate in sorted(lexItemUpdates.keys()):
        #print "probupdate is ",probupdate
        w = probupdate.split(" :: ")[0]
        if not wordstokeys.has_key(w): wordstokeys[w]=[(exp(lexItemUpdates[probupdate]),probupdate)]
        else: wordstokeys[w].append((exp(lexItemUpdates[probupdate]),probupdate))
    
    #print "\n\nLexUpdates"
    #for w in wordstokeys:
    #    wordsum = 0.0
    #    for p in reversed(sorted(wordstokeys[w])):
    #        wordsum = wordsum+p[0]
    #        print p
    #    print "wordsum = ",wordsum
    #    if wordsum<1.0: print "wordsum < 1 for ",w,wordsum
    #    if wordsum>1.0: print "wordsum > 1 for ",w,wordsum
        
    #for probupdate in sorted(lexItemUpdates.keys()):
        #print probupdate,"::",lexItemUpdates[probupdate]
    #print "\n\n"



    # got learning rate
    


    iteratetoconv = False # this should not be true with online learning 
    logLikelihoodDiff = abs((norm_score-old_norm)/norm_score)
    #print "loglikelihooddiff between ",norm_score," and ",old_norm," is ",logLikelihoodDiff
    if (doupdates and logLikelihoodDiff > 0.05 and iteratetoconv):
#        lexicon.set_temp_params()
#        RuleSet.set_temp_params()
        RuleSet.perform_temp_updates()
        lexicon.perform_temp_updates()
        RuleSet.clear_updates()
        lexicon.clear_updates()
        print ">",
        i_o_oneChart(chart,sem_store,lexicon,RuleSet,doupdates,norm_score)
   
    elif doupdates:
        learningrate = lexicon.get_learning_rate(sentence_count)
        #learningrate = 1.0/(sentence_count+1)
        print "doing updates with learning rate ",learningrate
        RuleSet.perform_updates(learningrate,datasize,sentence_count)
        lexicon.perform_updates(learningrate,datasize,sentence_count)
        #lexicon.print_all_word_probs(sentence_count,sem_store)
        #lexicon.set_temp_params()
        #RuleSet.set_temp_params()
    RuleSet.clear_updates()
    lexicon.clear_updates()
    return norm_score
        # for target in RulesUsed:
    # RuleSet.update_alphas()
    # lexicon.update_alphas()  
    
    #rc = RuleSet.compare_probs()
    #lc = lexicon.compare_probs()
    
    #RuleSet.clear_probs()
    #lexicon.clear_probs()

    #return rc
    #for item in LexItemsUsed:
        #lexicon.update_alpha(item[0],item[1],item[2])
    
    #for target in RulesUsed:
        #RuleSet.clear_probs(target)
    #for item in LexItemsUsed:
        #lexicon.clear_probs(item[0],item[1],item[2])
    
