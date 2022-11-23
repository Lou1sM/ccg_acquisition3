# inside-outside maths
from math import log
from math import exp
from tools import log_sum
from tools import log_diff
from tools import inf
import lexicon_classes

def i_o(sentence_charts, sem_store, lexicon, rule_set, old_log_prob):

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

def i_o_oneChart(chart,sem_store,lexicon,rule_set,doupdates,old_norm,\
                     sentence_count,generating=False,sentToGen=None):
    datasize = 4000
    gamma = 1.0/datasize
    RulesUsed = []
    LexItemsUsed = []
    wordFromSemSyn = {}
    semFromSyn = {}
    verbose = False

    # chart is dict of dicts, outer keys are ints (levels), e.g. 1,..,6, inner
    # keys are tuples (<category>, <lf>, int, int), values are chart_entries, each of which is some
    # representation of a sentence?
    for level in chart:
        for item in sorted(chart[level]):
            entry = chart[level][item]
            entry.word_score = lexicon.get_log_word_prob(entry.word_target, entry.syn_key, entry.sem_key, sentence_count)

            if not entry.word_score == -inf:
                if entry.sem_key+":"+entry.syn_key+":"+entry.word_target not in wordFromSemSyn:
                    wordFromSemSyn[entry.sem_key+":"+entry.syn_key+":"+entry.word_target] = entry.word_score
                if entry.sem_key+":"+entry.syn_key not in semFromSyn:
                    semFromSyn[entry.sem_key+":"+entry.syn_key] = entry.sem_score

            # should probably have a distinct node for the semantic part of this
            rule_score = rule_set.return_log_prob(entry.syn_key, entry.syn_key+'_LEX')
            entry.inside_score = entry.word_score+entry.sem_score+rule_score
            entry.addNumParses(1)

            rule_score = -inf
            for pair in entry.children:
                left_syn = pair[0][0]
                right_syn = pair[1][0]
                target = left_syn+'#####'+right_syn
                rule_score = rule_set.return_log_prob(entry.syn_key, target)
                entryL = chart[pair[0][3]-pair[0][2]][pair[0]]
                entryR = chart[pair[1][3]-pair[1][2]][pair[1]]
                new_inside_score = rule_score+entryL.inside_score+entryR.inside_score
                entry.inside_score = log_sum(entry.inside_score, new_inside_score)
                entry.addNumParses(entryL.getNumParses()*entryR.getNumParses())
                #print([(x,getattr(entry,x)) for x in dir(entry) if 'score' in x])


    if verbose:
        print("\n\ninside scores ")
        for level in chart:
            for item in chart[level]:
                entry = chart[level][item]
                print(entry.to_string(), "  ", entry.inside_score)
        print("\n\n")

    top_down = list(range(1, len(chart)+1))
    top_down.reverse()

    for level in top_down:
        #print "level top down is ",level
        for item in chart[level]:
            entry = chart[level][item]
            #print "entry is ",entry.to_string()
            usestartrule = True
            if len(entry.parents) == 0:
                topcat = chart[len(chart)][item].syn_key
                if usestartrule:
                    rule_set.check_start_rule(chart[len(chart)][item].ccgCat.syn)
                    entry.outside_score = rule_set.return_log_prob("START", topcat)
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
                    left_syn = pair[0][0]
                    right_syn = pair[1][0]
                    target = left_syn+'#####'+right_syn
                    #rule_prob = rule_set.return_prob(father.syn_key,target)
                    rule_score = rule_set.return_log_prob(father.syn_key, target)
                    s_in_prob = 0
                    if side == 'l':
                        s_in_score = chart[pair[1][3]-pair[1][2]][pair[1]].inside_score
                    elif side == 'r':
                        s_in_score = chart[pair[0][3]-pair[0][2]][pair[0]].inside_score
                    #entry.outside_prob += p_out_prob*rule_prob*s_in_prob
                    entry.outside_score = log_sum(entry.outside_score, p_out_score+s_in_score+rule_score)

    norm_score = -inf
    # should go up to where inside is done
    # Really do need to put a START node in here
    for item in chart[top_down[0]]:
        topcat = chart[top_down[0]][item].syn_key
        norm_score = log_sum(norm_score, chart[top_down[0]][item].inside_score+chart[top_down[0]][item].outside_score)

    if norm_score == -inf : return
    for item in chart[top_down[0]]:
        topcat = chart[top_down[0]][item].syn_key
        log_score_start = chart[top_down[0]][item].inside_score+chart[top_down[0]][item].outside_score - norm_score
        rule_set.store_log_update(topcat, log_score_start)


    lexItemUpdates = {}
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
                ##print "entry with scores is ",entry.to_string()," inside is ",entry.inside_score," outside is ",entry.outside_score
                #rule_score = rule_set.return_log_prob(entry.syn_key,entry.syn_key+'_LEX')
                #onewordprobs[entry.p] = log_sum(onewordprobs[i],entry.inside_score+entry.outside_score)

    onewordupdates = {}
    for i in range(len(chart)): onewordupdates[i]=-inf
    for level in chart:
        for item in chart[level]:
            entry = chart[level][item]
            #if entry.sentence == ['were','you', 'a', 'pirate','too','?']:
                #breakpoint()
            B_pq = entry.inside_score
            a_pq = entry.outside_score

            node_score = B_pq+a_pq

            node_sum = -inf
            node_inside_sum = -inf
            for pair in entry.children:
                l_child = chart[pair[0][3]-pair[0][2]][pair[0]]
                r_child = chart[pair[1][3]-pair[1][2]][pair[1]]
                left_syn = l_child.syn_key
                right_syn = r_child.syn_key

                target = left_syn+'#####'+right_syn
                # want the rule score
                rule_score = rule_set.return_log_prob(entry.syn_key, target)
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
                node_inside_sum = log_sum(node_inside_sum, child_score + rule_score)

                if logRuleExp >= 0.0:
                    print("logRuleExp = ", logRuleExp)
                    print("child score is ", child_score)
                    print("outside is ", a_pq)
                    print("rule score is ", rule_score)
                    print("norm score is ", norm_score)
                #print "logER is ",logER
                rule_set.store_log_update(target, logRuleExp)

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
                ##rule_set.update_target_p(a_pq,B_pd,B_dq,target)

                #rule_prob =
                #rule_set.update_target_p(a_pq,B_pd,B_dq,target)

                #if not target in RulesUsed:
                    #RulesUsed.append(target)
                ##################################
                ##################################
            if node_score==-inf:
                if len(entry.words)==1: print("inf for one word")
                continue

            target = entry.syn_key+'_LEX'
            rule_score = rule_set.return_log_prob(entry.syn_key, target)

            # really need to work out what this should be, but e^1e-5 is small
            # think about norm though

            if 1E-5>=entry.inside_score-node_inside_sum>=-1E-5:
            #if B_pq - node_inside_sum
                if entry.q-entry.p==1: print("lexscore is -inf for ", entry.lexKey())
                lex_score = -inf
            else:
                #lex_score = log_diff(node_score,node_sum)
                lex_score = log_diff(entry.inside_score, node_inside_sum)

            logLexExp = (lex_score + a_pq)  - norm_score

            if len(entry.words)==1:
                if verbose:
                    if node_inside_sum!=-inf: print("for ", entry.word_target, " insidesum is ", node_inside_sum)
                onewordupdates[entry.p]=log_sum(onewordupdates[entry.p], logLexExp)

            #logLexExp = logLexExp  - norm_score
            #target = entry.syn_key+'_LEX'
            #lex_prob = rule_set.return_prob(entry.syn_key,target)
            #lex_score = rule_set.return_log_prob(entry.syn_key,target)
            #word_prob = lexicon.get_word_prob(entry.word_target,entry.syn_key,entry.sem_key)
            #word_score = lexicon.get_log_word_prob(entry.word_target,entry.syn_key,entry.sem_key)
            #logER = lex_score + word_score
            #rule_set.update_target_log_p(a_pq,1.0,word_prob,target)
            #rule_set.update_target_log_p(logER,target)
            if logLexExp >= 0.0:
                print("cell is ", entry.to_string())
                print("entry inside is ", entry.inside_score)
                print("inside sum is ", node_inside_sum)

                print("lex score  = ", lex_score)

                #print "child score is ",child_score
                print("outside is ", a_pq)
                #print "rule score is ",rule_score
                print("norm score is ", norm_score)
                #lex_score = log_diff(entry.inside_score,node_inside_sum)

            rule_set.store_log_update(target, logLexExp)
            # this only needs to be done once for each time
            # the rule head is seen
            #rule_set.update_bottom(entry.syn_key,a_pq,B_pq)
            #
            lexicon.store_log_update(entry.word_target, entry.syn_key, entry.sem_key, logLexExp)
            if entry.words ==['too','?']:
                print(lex_score)
            if lex_score != -inf:
                if len(entry.words)>1 and lexicon_classes.syn_sem.one_word:
                    print("\nERROR, NONZERO UPDATE FOR MWE ", entry.lexKey())
                    print("node inside ", entry.inside_score)
                    print("node score ", node_score)
                    print("norm score ", norm_score)
                    print("node inside sum ", node_inside_sum)
                    print("diff is ", entry.inside_score-node_inside_sum)
                    print("lex score ", lex_score, "\n")
                    print("lex alpha ", lexicon.lex[(entry.word_target, entry.syn_key, entry.sem_key)].alpha)
                if entry.lexKey() not in lexItemUpdates:
                    lexItemUpdates[entry.lexKey()] = logLexExp
                else:  lexItemUpdates[entry.lexKey()] += logLexExp
            elif len(entry.words)==1:
                print("\nlex score is -inf for ", entry.lexKey())


            #entry.sem_prob = lexicon.get_sem_prob(entry.syn_key,entry.sem_key,sem_store)
            #if not (entry.word_target,entry.cat.key,entry.sem_key) in LexItemsUsed:
            #LexItemsUsed.append((entry.word_target,entry.cat.key,entry.sem_key))

    #for i in range(len(chart)): print "update sum for ",i," is ",onewordupdates[i]
    #if not entry.word_score == -inf:
    #print "\n\nLexProbs:"
    #print "wordProbs"
    #sortedl = []

    wordstokeys = {}
    for probupdate in sorted(lexItemUpdates.keys()):
        w = probupdate.split(" :: ")[0]
        if w not in wordstokeys: wordstokeys[w]=[(exp(lexItemUpdates[probupdate]), probupdate)]
        else: wordstokeys[w].append((exp(lexItemUpdates[probupdate]), probupdate))

    iteratetoconv = False # this should not be true with online learning
    logLikelihoodDiff = abs((norm_score-old_norm)/norm_score)
    if (doupdates and logLikelihoodDiff > 0.05 and iteratetoconv):
        rule_set.perform_temp_updates()
        lexicon.perform_temp_updates()
        rule_set.clear_updates()
        lexicon.clear_updates()
        print(">", end=' ')
        i_o_oneChart(chart, sem_store, lexicon, rule_set, doupdates, norm_score)

    elif doupdates:
        learningrate = lexicon.get_learning_rate(sentence_count)
        rule_set.perform_updates(learningrate, datasize, sentence_count)
        lexicon.perform_updates(learningrate, datasize, sentence_count)
    rule_set.clear_updates()
    lexicon.clear_updates()
    return norm_score
