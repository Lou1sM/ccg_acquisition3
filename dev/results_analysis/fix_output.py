import sys, pylab, pdb, os
import numpy as np
import matplotlib.pyplot as plt
import extract_from_lexicon
from optparse import OptionParser


FIELD_NAMES_LEX = ['name', 'sentence count', 'row type', 'LF', 'word', 'prob']
FIELD_NAMES_SYN = ['name', 'sentence count', 'syn_cat', 'row_type', 'prob']
REGARD_TWO_LFS = True

W_GIVEN_LF1 = 0
W_GIVEN_LF2 = 1
LF1 = 2
LF2 = 3
SYN_GIVEN_LF1 = 4
SYN_GIVEN_LF2 = 5
SYN_GIVEN_LF1_OR_LF2 = 6
W_GIVEN_LF1_OR_LF2 = 7

NUM_FIELDS = 8

def recompute_transitives(filenames):
    sc_to_syns = {}
    sc_words = {}
    seen_scs = set()
    for fn in filenames:
        f = open(fn)
        for line in f:
            line = line.strip()
            if 'Transitives' in line:
                if 'Pr(syn|all relevant syns)' in line:
                    fields = dict(zip(FIELD_NAMES_SYN,line.split('\t')))
                    sc_to_syns[(fields['sentence count'],fields['syn_cat'])] = float(fields['prob'])
                    seen_scs.update(fields['sentence count'])
                else:
                    fields = dict(zip(FIELD_NAMES_LEX,line.split('\t')))
                    key = (fields['sentence count'],fields['word'],fields['row_type'])
                    sc_words[key] = float(prob)
                    seen_scs.update(fields['sentence count'])
    
    # fix syn|LF1 or LF2
    for sc,word,row_type in sc_words:
        if row_type == 'Pr(correct syn|LF1 or LF2)':
            pr_lf1 = sc_words[(sc,word,'Pr(LF1)')]
            pr_lf2 = sc_words[(sc,word,'Pr(LF2)')]
            pr_syn_given_lf1 = sc_words[(sc,word,row_type)] * (pr_lf1 + pr_lf2) / pr_lf1
            mod_pr_lf2 = pr_lf2 



