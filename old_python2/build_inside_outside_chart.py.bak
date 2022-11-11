# Need to build inside-outside charts
# slight difference from usual because, we don't have 
# one to one word, sentence mappings. but that's ok.
# Hold as dictionary?
#     (head,p,q)

import pdb, re
# from exp import exp
# from variable import variable
# from lexicon_classes import syn_cat
from tools import inf
import expFunctions


########################################
# MAKE A CLASS CALLED SEM_STORE TO SAVE#
# ALL THE SEM REPS SO THAT THEY'RE NOT #
# NEEDED ALL OVER THE PLACE.           #
########################################


class SemStore:
    def __init__(self):
        self.store = {}

    def clear(self):
        self.store = {}

    def check(self, sem_key):
        if not isinstance(sem_key, str):
            raise (StandardError('key isnt string'))

        if not self.store.has_key(sem_key):
            return False
        else:
            return True

    def get_log_prior(self, sem_key):
        if not self.check(sem_key):
            print "not got ", sem_key
            sem = expFunctions.makeExpWithArgs(sem_key, {})
            # OMRI ADDED THE NEXT TWO LINES
            if isinstance(sem, tuple):
                sem = sem[0]
        else:
            sem = self.store[sem_key]
        return sem.semprior()

    def add(self, sem):
        self.store[sem.toString(True)] = sem
        if not self.store.has_key(sem.toStringShell(True)):
            self.store[sem.toStringShell(True)] = sem.makeShell({})

    def get(self, sem_key):
        if self.store.has_key(sem_key):
            return self.store[sem_key]
        else:
            return None


        ########################################
        # Each node for each span must be      #
        # realisable as a word.                #
        ########################################
        # class word_target:
        # def __init__(self,word):
        # self.word = word
        # self.inside_prob = 0.0
        # self.outside_prob = 0.0


########################################

########################################
# chart entries are for syntactic nodes#
# that represent the span p->q of the  #
# sentence.                               #
########################################
class chart_entry:
    def __init__(self, ccgCat, p, q, sentence):
        self.sentence = sentence
        if p is not None and q is not None:
            self.words = sentence[p:q]
        else:
            self.words = sentence
        self.ccgCat = ccgCat
        self.syn_key = ccgCat.synString()
        self.sem_key = ccgCat.semString()
        self.p = p
        self.q = q
        self.word_prob = 0.0
        self.word_score = -inf
        self.sem_prob = 0.0
        self.sem_score = -inf
        self.inside_prob = 0.0
        self.inside_score = -inf
        self.max_score = -inf
        self.outside_prob = 0.0
        self.outside_score = -inf
        self.children = []
        if p is not None and q is not None:
            self.word_target = ' '.join(sentence[p:q])
        else:
            self.word_target = ' '.join(sentence)
        self.parents = []
        self.numParses = 0

    def lexKey(self):
        return self.word_target + " :: " + self.syn_key + " :: " + self.sem_key

    def toString(self):
        return str(self.p) + ":" + str(
            self.q) + " :: " + self.word_target + " :: " + self.syn_key + " :: " + self.sem_key

    def add_parent(self, parent, side):
        instance = len(parent.children)
        self.parents.append((parent, instance, side))

    def addNumParses(self, np):
        self.numParses += np

    def getNumParses(self):
        return self.numParses

    def add_child(self, child):
        self.children.append(child)

    def clear_probs(self):
        self.word_prob = 0.0
        self.sem_prob = 0.0
        self.inside_prob = 0.0
        self.outside_prob = 0.0

    def get_inside(self):
        return self.inside_prob


def expand_chart(entry, chart, catStore, sem_store, RuleSet, lexicon, oneWord, correct_index):
    """
    CatStore is a dictionary that maps pairs of syntactic and semantic forms
    to the set of pairs they can decompose to. It's a cache essentially.
    """

    if entry.ccgCat.sem.getIsNull(): return
    if entry.p < entry.q - 1:
        for d in range(entry.p + 1, entry.q):
            words_l = ' '.join(entry.sentence[entry.p:d])
            words_r = ' '.join(entry.sentence[d:entry.q])
            if oneWord:
                minL = d - entry.p
                minR = entry.q - d
            else:
                minL = 0
                minR = 0
            # there is currently no support for null categories #
            for pair in entry.ccgCat.allPairs(catStore):
                l_cat = pair[0]
                l_syncat = l_cat.syn
                l_syn = l_cat.synString()
                l_sem = l_cat.semString()

                r_cat = pair[1]
                r_syncat = r_cat.syn
                r_syn = r_cat.synString()
                r_sem = r_cat.semString()

                direction = pair[2]
                numcomp = pair[3]

                if not sem_store.check(l_sem): sem_store.add(l_cat.sem)
                if not sem_store.check(r_sem): sem_store.add(r_cat.sem)

                # - this is needed to build an actual parse - #
                RuleSet.check_rule(entry.ccgCat.synString(), l_syncat, r_syncat, direction, numcomp)
                RuleSet.check_rule(l_syn, None, None, None, None)
                RuleSet.check_rule(r_syn, None, None, None, None)

                if correct_index:
                    lexicon.cur_cats.extend([r_cat, l_cat])

                lexicon.check(words_l, l_syn, l_sem, l_cat.sem)
                lexicon.check(words_r, r_syn, r_sem, r_cat.sem)

                # sem_store

                if not chart[d - entry.p].has_key((l_syn, l_sem, entry.p, d)):
                    cl = chart_entry(l_cat, entry.p, d, entry.sentence)
                    chart[d - entry.p][(l_syn, l_sem, entry.p, d)] = cl
                    # print "added entry ",(l_syn,l_sem,entry.p,d)
                    expand_chart(cl, chart, catStore, sem_store, RuleSet, lexicon, oneWord, correct_index)
                # else: print "already there"
                chart[d - entry.p][(l_syn, l_sem, entry.p, d)].add_parent(entry, 'l')
                if not chart[entry.q - d].has_key((r_syn, r_sem, d, entry.q)):
                    cr = chart_entry(r_cat, d, entry.q, entry.sentence)
                    chart[entry.q - d][(r_syn, r_sem, d, entry.q)] = cr
                    # print "added entry ",(r_syn,r_sem,d,entry.q)
                    expand_chart(cr, chart, catStore, sem_store, RuleSet, lexicon, oneWord, correct_index)
                # else: print "already there"
                chart[entry.q - d][(r_syn, r_sem, d, entry.q)].add_parent(entry, 'r')
                entry.add_child(((l_syn, l_sem, entry.p, d), (r_syn, r_sem, d, entry.q)))


########################################

########################################
# get_cats is called from get_children #
# and returns pairs of categories that #
# support the semantic decomposition.  #
# rl defines which side ('R','L') that #
# the head goes on                       #
########################################
"""
def get_cats(parent_cat,head_rep,s1,s2,minL,minR):
    child_pairs = []
    # pair structure is (cat, sem_string) # 
    # could just go to cat
    
    if len(s2.lambdas) == 0:
        # Functional application #
        # fwd
        # s1->L, s2->R
        if len(s1.all_deps)+1 >= minL and len(s2.all_deps)+1 >= minR:
            cat1 = parent_cat.copy()
            cat1.add_target(s2.type,'fwd')
            cat2 = syn_cat(s2.type,[])
            child_pairs.append(((cat1,s1.sem_key),(cat2,s2.sem_key)))
            #print "got fwd fa ", cat1.key,s1.sem_key,"  ",cat2.key,s2.sem_key
        # back
        if len(s2.all_deps)+1 >= minL and len(s1.all_deps)+1 >= minR:
            cat1 = parent_cat.copy()
            cat1.add_target(s2.type,'back')    
            cat2 = syn_cat(s2.type,[])
            child_pairs.append(((cat2,s2.sem_key),(cat1,s1.sem_key)))
            #print "got back fa ", cat2.key,s2.sem_key,"  ",cat1.key,s1.sem_key
    elif len(s2.lambdas) + len(s1.lambdas) == len(head_rep.lambdas)+1:
        # Go straight for generalised composition
        # check that this is cool
        no_rem = len(s2.lambdas)
        # fwd
        if len(s1.all_deps)+1 >= minL and len(s2.all_deps)+1 >= minR:
            cat1 = parent_cat.copy()
            extension = cat1.remove_targets(no_rem)
            # what about complex things? e.g s/(s\np) (s\np)/np #
            cat1.add_target(s2.type,'fwd')
            cat2 = syn_cat(s2.type,[])
            cat2.extend_targets(extension)
            child_pairs.append(((cat1,s1.sem_key),(cat2,s2.sem_key)))
            #print "got fwd comp ", cat1.key,s1.sem_key,"  ",cat2.key,s2.sem_key
        # back
        if len(s2.all_deps)+1 >= minL and len(s1.all_deps)+1 >= minR:
            cat1 = parent_cat.copy()
            extension = cat1.remove_targets(no_rem)
            cat1.add_target(s2.type,'back')
            cat2 = syn_cat(s2.type,[])
            child_pairs.append(((cat2,s2.sem_key),(cat1,s1.sem_key)))
            #print "got back comp ", cat2.key,s2.sem_key,"  ",cat1.key,s1.sem_key
    return child_pairs

########################################
def build_chartOld(sem_list,sentence,sem_store,RuleSet,lexicon,oneWord):
    chart = {}
    for i in range(1,len(sentence)+1):
                chart[i] = {}
    print 'sentence is ',sentence
    for sem in sem_list:    
        print 'sem is ',sem
    # NEED START SYMBOL
        cat = syn_cat(sem_store.get(sem).type,[])
        c1 = chart_entry(sem,cat,0,len(sentence),sentence)
        chart[len(sentence)][(cat.key,sem,0,len(sentence))] = c1
        RuleSet.check_rule(c1.cat.key,None,None)
        wordspan = ' '.join(sentence)
        lexicon.check(wordspan,c1.cat.key,c1.sem_key,sem)
        
        expand_chart(c1,chart,sem_store,RuleSet,lexicon,oneWord)
    chart_size = 0
    for level in chart:
        chart_size += len(chart[level])
    print 'size of chart is ',chart_size
    return chart
"""


######################################################
# topCat,sentence,RuleSet,lexicon,sem_store,oneWord) #
######################################################

def build_chart(topCatList, sentence, RuleSet, lexicon, catStore, sem_store, oneWord):
    chart = {}
    for i in range(1, len(sentence) + 1):
        chart[i] = {}
    print 'sentence is ', sentence
    # for sem in sem_list:
    # NEED START SYMBOL
    # cat = syn_cat(sem_store.get(sem).type,[])
    # need to fix outside probs too!
    correct_index = (len(topCatList) - 1) / 2  # the index of the correct semantics
    for ind, topCat in enumerate(topCatList):
        print 'sem is ', topCat.sem.toString(True)
        c1 = chart_entry(topCat, 0, len(sentence), sentence)
        if not sem_store.check(topCat.sem.toString(True)):
            sem_store.add(topCat.sem)
        chart[len(sentence)][(topCat.synString(), topCat.semString(), 0, len(sentence))] = c1
        RuleSet.check_start_rule(topCat.syn)
        RuleSet.check_rule(topCat.synString(), None, None, None, None)
        wordspan = ' '.join(sentence)  # ,topCat.synString(),topCat.semString())
        lexicon.check(wordspan, topCat.synString(), topCat.semString(), topCat.sem)
        expand_chart(c1, chart, catStore, sem_store, RuleSet, lexicon, oneWord, correct_index == ind)

    chart_size = 0
    for level in chart:
        chart_size += len(chart[level])
    print 'size of chart is ', chart_size
    return chart
