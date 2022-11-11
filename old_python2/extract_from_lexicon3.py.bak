"""
Reads a pickled lexicon and computes the distributions according to that lexicon.
"""
import parser, SemLearn, generator, exp, cat, pdb, os
import scipy.misc
import inside_outside_calc
import cPickle as pickle
import sys, math, re, os
import lexicon_classes
import numpy as np
from cat import synCat

ADVERBS = ["now","again","on","later","away","off","almost","yet","hard","first","already",\
               "better","still","maybe","well",\
               "fine","fast","right","before","round","quick","far","ever","either",\
               "anymore","along","alone","ahead","whole",\
               "together","though","straight","soon","sometimes","sideways","perhaps","once","next","never",\
               "long","instead","good","barefoot","awhile","as","apart","afterwards"]
AUXILIARIES = ["do", "did", "does", "done", "is", "was", "were", "been", \
                   "are", "be", "am", "have", "has", "had"]
WH_WORDS = ["what"]
non_slashed_syn_types = True

swap_args = lambda s: re.sub('\\(\\$1,\\$0,\\$2\\)','($0,$1,$2)',s)
swap_args_other_way = lambda s: re.sub('\\(\\$0,\\$1,\\$2\\)','($1,$0,$2)',s)
PATH = os.path.dirname(os.path.realpath(__file__)) + '/'

######################################################
# EXTRACTING THE SYNTAX DISTRIBUTION
######################################################

def _catProbFromGrammar(sc,RuleSet):
    """
    obtains Pr(syntactic category) 
    """
    c_prob = 1.0
    rule_prob = 1.0
    if sc.atomic(): return 1.0
    elif sc.direction=="fwd":
        # fwd
        rule = (sc.funct.toString(),sc.toString()+'#####'+sc.arg.toString())
        if RuleSet.check_target(rule[1]):
            rule_prob = RuleSet.return_prob(rule[0],rule[1])
            #print "rule_prob is ",rule_prob
        #else:
        #    print 'not got rule for ',rule[1]
        c_prob = c_prob*rule_prob
        c_prob = c_prob*_catProbFromGrammar(sc.funct,RuleSet)
    elif sc.direction=="back":
        rule = (sc.funct.toString(),sc.arg.toString()+'#####'+sc.toString())
        if RuleSet.check_target(rule[1]):
            rule_prob = RuleSet.return_prob(rule[0],rule[1])
            #print "rule_prob is ",rule_prob
        #else:
        #    print 'not got rule for ',rule[1]
        c_prob = c_prob*rule_prob
        c_prob = c_prob*_catProbFromGrammar(sc.funct,RuleSet)

    return c_prob

def get_synt_distribution(target_syn_keys, lexicon, sem_store, RuleSet, sentence_count):
    """
    returns the distribution over the syntactic categories in target_syn_keys (inputted as strings).
    """
    synt_cat_distribution = []
    for syn_key in target_syn_keys:
        pr_syn = _catProbFromGrammar(synCat.readCat(syn_key),RuleSet)    # an approximation of Pr(cat)
        leaf_log_prob =  RuleSet.return_leaf_map_log_prob(syn_key) # added 28/7
        if leaf_log_prob is None:
            return {}
        leaf_prob = math.exp(leaf_log_prob)
        synt_cat_distribution.append((syn_key,pr_syn*leaf_prob))
    S = sum([x[1] for x in synt_cat_distribution])
    if S > 0:
        synt_cat_distribution = dict([(k,v/S) for k,v in synt_cat_distribution])
        return synt_cat_distribution
    else:
        return {}

def print_syn_yield_distribution(prefix,RuleSet,syn_key,sentence_count,f_out):
    """prints the 10 leading categories in the distribution of the yield of syn_key"""
    D = RuleSet.return_map_prob_distribution(syn_key)[:10]
    for cat,prob in D:
        f_out.write('\t'.join([prefix,str(sentence_count),'Pr(yield syn|syn)',\
                                   syn_key, cat[0],(cat[1] if len(cat) > 1 else 'EMPTY'),str(prob)])+'\n')

######################################################
# EXTRACTING THE DISTRIBUTION OF LOGICAL FORMS
######################################################
def get_w_dist_given_LF(lexicon,sentence_count,sem_store,RuleSet,target_sem_key):
    """
    computing the distribution Pr(w=*|LF)
    """
    all_words = [w[1] for w in lexicon.sem_to_word.keys() if w[0] == target_sem_key]
    sem_types = lexicon.sem_distribution.types_for_sem(target_sem_key)
    if len(sem_types) == 0: # if the LF has never been seen before
        return {}
    else:
        sem_type = sem_types[0]
    w_dist_given_LF = {}
    Z = 0.0
    for target_word in all_words:
        log_pr = lexicon.get_map_log_word_prob(target_word,sem_type,target_sem_key,sentence_count)
        if log_pr is None:
            continue
        pr_w_given_sem = math.exp(log_pr)
        w_dist_given_LF[target_word] = pr_w_given_sem
        Z += pr_w_given_sem
    w_dist_given_LF = dict([(k,v/Z) for k,v in w_dist_given_LF.items()])
    return w_dist_given_LF

def print_lf_given_word_probs(phenom_name,lexicon,target_word,sem_store,RuleSet,sentence_count,f_out):
    """prints only the distribution of Pr(LF|w) based on pseudo-counts"""
    lf_given_word_D = get_lf_given_word_probs(lexicon,target_word,sem_store,RuleSet,sentence_count)
    lf_prob_pairs = sorted(lf_given_word_D.items(),key=lambda x:-x[1])[:10]
    for lf_prob_pair in lf_prob_pairs:
        f_out.write('\t'.join([phenom_name,str(sentence_count),'Pr(correct LF|w)',\
                                   lf_prob_pair[0],str(target_word),str(lf_prob_pair[1])])+'\n')

def print_syn_lf_given_word_probs(phenom_name,lexicon,target_word,sem_store,RuleSet,sentence_count,f_out):
    """prints only the distribution of Pr(LF|w) based on pseudo-counts"""
    lf_given_word_D = get_lf_given_word_probs(lexicon,target_word,sem_store,RuleSet,sentence_count,also_syn=True)
    lf_prob_pairs = sorted(lf_given_word_D.items(),key=lambda x:-x[1])[:10]
    for lf_prob_pair in lf_prob_pairs:
        f_out.write('\t'.join([phenom_name,str(sentence_count),'Pr(correct syn,LF|w)',\
                                   lf_prob_pair[0][0],lf_prob_pair[0][1],\
                                   str(target_word),str(lf_prob_pair[1])])+'\n')

def get_lf_given_word_probs(lexicon,target_word,sem_store,RuleSet,sentence_count,also_syn=False):
    """Tom's original code for getting the distribution Pr(LF=*|w)"""
    semdict = {}
    for l in lexicon.get_lex_items(target_word):
        if l.is_shell_item:
            continue
        lexicon.refresh_params(l.word,l.syn,l.sem_key,sentence_count)
        if not semdict.has_key(l.sem_key):
            log_pr_word_given_lf = lexicon.get_map_log_word_prob(target_word,l.syn,l.sem_key,sentence_count)
            log_pr_lf = lexicon.get_map_log_sem_prob(l.syn,l.sem_key,sem_store)
            if also_syn:
                semdict[(l.syn,l.sem_key)] = math.exp(log_pr_word_given_lf + log_pr_lf)
            else:
                semdict[l.sem_key] = math.exp(log_pr_word_given_lf + log_pr_lf)
    semalphatot = sum(semdict.values())
    if semalphatot==0.0:
        return {}
    semdict = dict([(k,v/semalphatot) for k,v in semdict.items() if v > 0])
    return semdict

def get_synt_given_LF(lexicon,sentence_count,sem_store,RuleSet,target_syn_keys,\
                          target_sem_key,synt_cat_distribution):
    """
    computing Pr(syn|LF).
    target_syn_keys: the list of relevant syntactic categories.
    target_sem_key: the target LF
    sem_store: probably a mapping between strings of lambda expresions and lambda expression objects.
    """
    Z = 0.0
    syn_given_sem_D = []
    for syn_key in synt_cat_distribution.keys():
        pr_sem_given_syn = math.exp(lexicon.get_map_log_sem_prob(syn_key,target_sem_key,sem_store))
        Z += synt_cat_distribution[syn_key] * pr_sem_given_syn
        syn_given_sem_D.append((syn_key,synt_cat_distribution[syn_key] * pr_sem_given_syn ))
    syn_given_sem_D = dict([(k,v/Z) for k,v in syn_given_sem_D])
    
    return syn_given_sem_D, synt_cat_distribution


def get_transitive_cats(lexicon,sentence_count,sem_store,RuleSet,syn_cat_distribution):
    """
    Returns all the 8 transtive categories.
    """
    syn_key  = '((S|NP)|NP)'
    transitive_shells = [k[1] for k in lexicon.sem_distribution.type_shell_to_count.keys() \
                             if k[0] == syn_key]
    L_pr_aligned = []
    L_pr_notaligned = []
    for sh in transitive_shells:
        ov = orderOfVariables(sh)
        if ov is None:
            continue
        log_pr = \
            lexicon.get_log_shell_given_type_prob(syn_key,sh,sem_store,sentence_count)
        if ov:
            L_pr_aligned.append(log_pr)
        else:
            L_pr_notaligned.append(log_pr)
    log_pr_aligned = scipy.misc.logsumexp(L_pr_aligned)
    log_pr_notaligned = scipy.misc.logsumexp(L_pr_notaligned)
    pr_aligned = math.exp(log_pr_aligned) / \
        (math.exp(log_pr_aligned) + math.exp(log_pr_notaligned))
    pr_notaligned = 1 - pr_aligned
    output = {}
    output['SVO'] = syn_cat_distribution['((S\\NP)/NP)']
    output['weird SVO'] = 0.0
    output['OVS'] = syn_cat_distribution['((S/NP)\\NP)']
    output['weird OVS'] = 0.0
    output['VSO'] = syn_cat_distribution['((S/NP)/NP)'] * pr_aligned
    output['VOS'] = syn_cat_distribution['((S/NP)/NP)'] * pr_notaligned
    output['OSV'] = syn_cat_distribution['((S\\NP)\\NP)'] * pr_aligned
    output['SOV'] = syn_cat_distribution['((S\\NP)\\NP)'] * pr_notaligned
    return output

def get_adjective_order(lexicon,sentence_count,sem_store,RuleSet,syn_cat_distribution):
    """
    Returns all the 8 transtive categories.
    """
    syn_key  = '((NP|NP)|NP)'
    transitive_shells = [k[1] for k in lexicon.sem_distribution.type_shell_to_count.keys() \
                             if k[0] == syn_key]
    L_pr_aligned = []
    L_pr_notaligned = []
    for sh in transitive_shells:
        ov = orderOfVariables(sh)
        if ov is None:
            continue
        log_pr = \
            lexicon.get_log_shell_given_type_prob(syn_key,sh,sem_store,sentence_count)
        if ov:
            L_pr_aligned.append(log_pr)
        else:
            L_pr_notaligned.append(log_pr)
    log_pr_aligned = scipy.misc.logsumexp(L_pr_aligned)
    log_pr_notaligned = scipy.misc.logsumexp(L_pr_notaligned)
    pr_aligned = math.exp(log_pr_aligned) / \
        (math.exp(log_pr_aligned) + math.exp(log_pr_notaligned))
    pr_notaligned = 1 - pr_aligned
    output = {}
    output['SVO'] = syn_cat_distribution['((S\\NP)/NP)']
    output['weird SVO'] = 0.0
    output['OVS'] = syn_cat_distribution['((S/NP)\\NP)']
    output['weird OVS'] = 0.0
    output['VSO'] = syn_cat_distribution['((S/NP)/NP)'] * pr_aligned
    output['VOS'] = syn_cat_distribution['((S/NP)/NP)'] * pr_notaligned
    output['OSV'] = syn_cat_distribution['((S\\NP)\\NP)'] * pr_aligned
    output['SOV'] = syn_cat_distribution['((S\\NP)\\NP)'] * pr_notaligned
    return output

def orderOfVariables(sem_str):
    """Returns true if the order of appearance is the same as the order of binding. False otherwise."""
    all_vars = re.findall('\\$[0-9]',sem_str)
    D = {}
    for ind,v in enumerate(all_vars):
        cur = D.get(v,[])
        cur.append(ind)
        D[v] = cur
    L = sorted(D.values(),key=lambda x: x[0])
    if len(L) < 2 or len(L[0]) < 2 or len(L[1]) < 2:
        return None
    else:
        return L[0][1] < L[1][1]


######################################################
# PHENOMENA
######################################################

class Phenomenon:
    
    def __init__(self, name, word_lf_pairs, target_syn, all_target_syns, \
                     unambig = None, two_LFs = False, \
                     flipped_target_syn=None,sem_type=None, \
                     target_shells=None):
        self._name = name
        self._word_LF = set(word_lf_pairs) # a set of correct (word,LF) pairs
        self._target_syn = target_syn
        self._all_target_syns = set(all_target_syns)
        self._two_LFs = two_LFs # whether to consider the flipped LF or not
        self._flipped_target_syn = flipped_target_syn
        if unambig is None:
            self._unambig = self.target_words()
        else:
            self._unambig = unambig
        self._sem_type = sem_type
        self._target_shells = target_shells

    def sem_type(self):
        return self._sem_type

    def target_types_shells(self):
        if self._target_shells is not None:
            return [(self._sem_type,shell) for shell in self._target_shells]
        else:
            return []

    def target_syn_shells(self):
        """
        the same as target_types_shells but for the case where syn_to_type is the identity function.
        """
        if self._target_shells is not None:
            return [(syn_key,shell) for shell in self._target_shells for syn_key in self._all_target_syns]
        else:
            return []
        
    def two_LFs(self):
        return self._two_LFs
    
    def name(self):
        """the name of the phenomenon"""
        return self._name
    
    def target_lf(self,w):
        L = [x[1] for x in self._word_LF if x[0] == w]
        if len(L) > 1:
            return None #ambiguous verb
        elif L == []:
            raise Exception('Incorrectly defined Phenomenon')
        return L[0]

    def target_lfs(self):
        return [x[1] for x in self._word_LF]

    def target_syn(self):
        return self._target_syn

    def flipped_target_syn(self):
        return self._flipped_target_syn
        
    def all_target_syns(self):
        """all distractor syns"""
        return self._all_target_syns
    
    def target_word(self,lf):
        """the correct word that matches this lf"""
        other_lf = swap_args_other_way(lf)
        if lf == other_lf:
            L = [x[0] for x in self._word_LF if x[1] == lf]
        else:
            L = [x[0] for x in self._word_LF if x[1] == lf]
            L.extend([x[0] for x in self._word_LF if x[1] == other_lf])
        if len(L) != 1:
            print('Incorrectly defined Phenomenon')
            print("\n\n*******" + lf + "*******\n\n")
            # raise Exception('Incorrectly defined Phenomenon')
            return ''
        return L[0]
    
    def target_words(self):
        """all target words"""
        return [x[0] for x in self._word_LF]

    def unambig_words(self):
        return self._unambig

def get_phenomena():
    phenomena = []

    # adding transitives
    word_lfs = get_transitive_lfs()
    phenomena.append(Phenomenon('Transitives', word_lfs, '((S\NP)/NP)', \
                                    ["((S\NP)/NP)", "((S/NP)/NP)", "((S\NP)\NP)", "((S/NP)\NP)"], \
                                    two_LFs = True, \
                                    flipped_target_syn = "((S/NP)\NP)",sem_type='((S|NP)|NP)',\
                                    target_shells=['lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.placeholderP($0,$1,$2)',\
                                                       'lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.placeholderP($1,$0,$2)']))
    
    # adding intransitive verbs
    word_lfs = get_intransitive_lfs()
    phenomena.append(Phenomenon('Intransitives', word_lfs, '(S\NP)', \
                                    ['(S\NP)', "(S/NP)"],sem_type='(S|NP)'))
    
    # getting adjectives
    word_lfs = get_adjective_lfs()
    phenomena.append(Phenomenon('Adjectives', word_lfs, '(N/N)', \
                                    ['(N\N)', '(N/N)'],sem_type='(N|N)'))
    
    # getting prepositions
    word_lfs = get_prep_lfs()
    #phenomena.append(Phenomenon('Prepositions', word_lfs, '(PP/NP)', \
    #                                ['(PP/NP)', "(PP\NP)"],sem_type='(PP|NP)'))
    phenomena.append(Phenomenon('Prepositions', word_lfs, '((NP\NP))/NP)', \
                               ['((NP/NP)/NP)', '((NP\NP)\NP)', '((NP\NP)/NP)', '((NP/NP)\NP'], sem_type='((NP|NP)|NP)'))    

    # getting determiners
    word_lfs = get_det_lfs()
    phenomena.append(Phenomenon('Determiners', word_lfs, '(NP/N)', \
                                    ['(NP/N)', "(NP\N)"],sem_type='(NP|N)'))
    
    # getting nouns
    word_lfs = get_noun_lfs()
    phenomena.append(Phenomenon('Nouns', word_lfs, 'N', \
                                    ['N'],sem_type='N'))
    
    return phenomena

def get_bax_phenom():
    word_lfs = [('bax', 'lambda $0_{<e,t>}.lambda $1_{e}.and(adj|bax($1),$0($1))')]
    return Phenomenon('Adjectives', word_lfs, '(N\N)', \
                                    ['(N\N)', "(N/N)"],sem_type='(N|N)')
    
def get_dax_phenom():
    word_lfs = [('daxed', 'lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.v|dax&PAST($1,$0,$2)')]
    return Phenomenon('Daxed (trans.)', word_lfs, '((S\NP)/NP)', \
                          ['((S\NP)/NP)', "((S/NP)/NP)", "((S\NP)\NP)", "((S/NP)\NP)"],\
                          ['daxed'],\
                          two_LFs = True,\
                          flipped_target_syn = '((S/NP)\NP)',sem_type='((S|NP)|NP)',\
                          target_shells=['lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.placeholderP($0,$1,$2)',\
                                             'lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.placeholderP($1,$0,$2)'])

def get_dax2_phenom():
    word_lfs = [('dax', 'lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.v|dax($1,$0,$2)')]
    return Phenomenon('Dax2 (trans.)', word_lfs, '((S\NP)/NP)', \
                          ['((S\NP)/NP)', "((S/NP)/NP)", "((S\NP)\NP)", "((S/NP)\NP)"],\
                          ['dax'],\
                          two_LFs = True,\
                          flipped_target_syn = '((S/NP)\NP)',sem_type='((S|NP)|NP)',\
                          target_shells=['lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.placeholderP($0,$1,$2)',\
                                             'lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.placeholderP($1,$0,$2)'])

def get_acorp_phenom():
    word_lfs = [('corp', "lambda $0_{e}.lambda $1_{r}.prep|corp($0,$1)")]
    return Phenomenon('Acorp Prepositions', word_lfs, '(PP/NP)', \
                          ['(PP/NP)', "(PP\NP)"], sem_type='(PP|NP)')

def get_corp_phenom():
    word_lfs = [('corp', 'lambda $0_{e}.n|corp($0)')]
    return Phenomenon('Corp Nouns', word_lfs, 'N', \
                          ["N"], sem_type='N')

def get_gax_phenom():
    word_lfs = [('gaxed', 'lambda $0_{e}.lambda $1_{r}.v|gax&PAST($0,$1)')]
    return Phenomenon('Gaxed (intrans.)', word_lfs, '(S\NP)', \
                          ['(S\NP)', "(S/NP)"],sem_type='(S|NP)')

def get_jax_phenom():
    word_lfs = [('jax', 'lambda $0_{<e,t>}.det|jax($1,$0($1))')]
    return Phenomenon('Jax (deter.)', word_lfs, '(NP/N)', \
                                    ['(NP/N)', "(NP\N)"],sem_type='(NP|N)')

def get_pronoun_phenom():
    # getting pronouns
    word_lfs = [('you', 'pro|you'), ('it', 'pro|it'), ('I', 'pro|I')]
    return Phenomenon('Pronouns', word_lfs, 'NP', ['NP'],sem_type='NP')

def get_girl_phenom():
    word_lfs = [('girl', 'lambda $0_{e}.n|girl($0)')]
    return Phenomenon('Girl',word_lfs,'N',['N'],sem_type='N')

def get_transitive_lfs():
    verb_to_lf = []
    f = open(PATH+'logical_forms/verb_logical_forms')
    for line in f:
        fields = line.strip().split(' ')
        for k in ["lambda $0_{e}.lambda $1_{e}.lambda $2_{r}."+fields[1]+"($1,$0,$2)"]:
            verb_to_lf.append( (fields[0],k) )
    f.close()
    return verb_to_lf

def get_intransitive_lfs():
    verb_to_lf = []
    f = open(PATH+'logical_forms/verb_logical_forms')
    for line in f:
        fields = line.strip().split(' ')
        for k in ["lambda $0_{e}.lambda $1_{r}."+fields[1]+"($0,$1)"]:
            verb_to_lf.append( (fields[0],k) )
    f.close()
    return verb_to_lf

def get_adjective_lfs():
    verb_to_lf = []
    f = open(PATH+'logical_forms/adj_lfs')
    for line in f:
        fields = line.strip().split(' ')
        for k in ["lambda $0_{<e,t>}.lambda $1_{e}.and("+fields[1]+"($1),$0($1))"]:
            verb_to_lf.append( (fields[0],k) )
    f.close()
    return verb_to_lf

def get_prep_lfs():
    verb_to_lf = []
    f = open(PATH+'logical_forms/prep_lfs')
    for line in f:
        fields = line.strip().split(' ')
        #for k in ["lambda $0_{e}.lambda $1_{r}."+fields[1]+"($0,$1)"]:
        for k in ["lambda $0_{e}.lambda $1_{e}.att($1, " + fields[1] + "($0)"]:    
            verb_to_lf.append( (fields[0],k) )
    f.close()
    return verb_to_lf

def get_det_lfs():
    verb_to_lf = []
    f = open(PATH+'logical_forms/det_lfs')
    for line in f:
        fields = line.strip().split(' ')
        for k in ["lambda $0_{<e,t>}."+fields[1]+"($1,$0($1))"]:
            verb_to_lf.append( (fields[0],k) )
    f.close()
    return verb_to_lf

def get_noun_lfs():
    noun_to_lf = []
    f = open(PATH+'logical_forms/noun_lfs')
    for line in f:
        fields = line.strip().split(' ')
        k = "lambda $0_{e}."+fields[1]+"($0)"
        noun_to_lf.append( (fields[0],k) )
    f.close()
    return noun_to_lf


######################################################
# PUTTING EVERYTHING TOGETHER
######################################################

def log_map_sem_given_syn_prob(lexicon,syn_key,target_sem,sem_store):
    try:
        return math.exp(lexicon.get_map_log_sem_prob(syn_key,target_sem,sem_store))
    except Exception:
        return None

def print_cat_stats(phenom,lexicon,sentence_count,sem_store,RuleSet,f_out,prefix=''):
    """prints all relevant statistics for the phenomenon phenom"""
    synt_cat_distribution = get_synt_distribution(phenom.all_target_syns(), \
                                                      lexicon, sem_store, RuleSet, sentence_count)
    #if phenom.name() == 'Transitives':
    #    D = get_transitive_cats(lexicon,sentence_count,sem_store,RuleSet,synt_cat_distribution)
    #    for k,v in D.items():
    #        f_out.write(str(sentence_count)+'\t'+k+'\t'+str(v)+'\n')
    
    for syn_key in phenom.all_target_syns():
        f_out.write(prefix+'\t'.join([phenom.name(),str(sentence_count),syn_key,'Pr(syn|all relevant syns)',\
                                          str(synt_cat_distribution.get(syn_key,0.0))])+'\n')
    
    target_lfs = set(phenom.target_lfs())
    
    for target_lf in target_lfs:
        if not lexicon.sem_distribution.sem_to_pairs.has_key(target_lf):
            continue
        synt_given_LF, temp = get_synt_given_LF(lexicon,sentence_count,sem_store,RuleSet,\
                                                    phenom.all_target_syns(),target_lf,synt_cat_distribution)
        try:
            f_out.write(prefix+'\t'.join([phenom.name(),str(sentence_count),'Pr(correct syn|LF)',str(target_lf),\
                                          phenom.target_word(target_lf),\
                                          str(synt_given_LF[phenom.target_syn()])])+'\n')
        except KeyError:
            f_out.write(prefix+'\t'.join([phenom.name(),str(sentence_count),'Pr(correct syn|LF)',str(target_lf),\
                                          phenom.target_word(target_lf),\
                                          str(synt_given_LF[phenom.target_syn()])])+'\n')
        
        w_given_LF = get_w_dist_given_LF(lexicon,sentence_count,sem_store,RuleSet,target_lf)
        f_out.write(prefix+'\t'.join([phenom.name(),str(sentence_count),'Pr(correct word|LF)',str(target_lf),\
                                          phenom.target_word(target_lf),\
                                          str(w_given_LF.get(phenom.target_word(target_lf),0.0))])+'\n')
        
        if phenom.two_LFs():
            marginal_lf_probs = {}
            marginal_lf_probs[target_lf] = \
                math.exp(lexicon.get_map_log_sem_prob(phenom.target_syn(),target_lf,sem_store))
            pr_syn = 0.0
            for syn_key in ["((S/NP)/NP)", "((S\NP)\NP)"]:
                pr_syn += synt_cat_distribution.get(syn_key,0.0)
            other_lf = swap_args(target_lf)
            pr_other_lf = lexicon.get_map_log_sem_prob(phenom.target_syn(),other_lf,sem_store)
            if pr_other_lf is None:
                marginal_lf_probs[other_lf] = 0.0
            else:
                marginal_lf_probs[other_lf] = math.exp(pr_other_lf) * pr_syn
            
            for ind,target_sem in zip([1,2],[target_lf,other_lf]):
                f_out.write(prefix+'\t'.join([phenom.name(),str(sentence_count),'Pr(LF'+str(ind)+')',str(target_sem),\
                                                  phenom.target_word(target_lf),\
                                                  str(marginal_lf_probs.get(target_sem,0.0))])+'\n')
            
            prob_syn_given_lf1_or_lf2 = synt_given_LF[phenom.target_syn()] * marginal_lf_probs.get(target_lf) / \
                                             sum(marginal_lf_probs.values())
            
            target_word = phenom.target_word(target_lf)
            f_out.write(prefix+'\t'.join([phenom.name(),str(sentence_count),'Pr(correct syn|LF1 or LF2)',str(target_lf),\
                                              target_word,str(prob_syn_given_lf1_or_lf2)])+'\n')
            
            w_given_other_LF = get_w_dist_given_LF(lexicon,sentence_count,sem_store,RuleSet,other_lf)
            w_given_either_lf = (marginal_lf_probs[target_lf] * w_given_LF.get(target_word,0.0) + \
                                     marginal_lf_probs[other_lf] * w_given_other_LF.get(target_word,0.0)) / \
                                     sum(marginal_lf_probs.values())
            
            f_out.write(prefix+'\t'.join([phenom.name(),str(sentence_count),\
                                              'Pr(correct word|LF1 or LF2)',str(target_lf),\
                                              target_word,str(w_given_either_lf)])+'\n')

            
    for target_word in phenom.target_words():
        print_lf_given_word_probs(phenom.name(),lexicon,target_word,sem_store,RuleSet,sentence_count,f_out)
    
    sem_types_enabled = True
    for sem_type,shell_key in phenom.target_types_shells():
        try:
            pr = math.exp(lexicon.sem_distribution._log_shell_given_type_prob(sem_type,shell_key,\
                                                                                  sem_store,sentence_count,'log'))
            f_out.write(prefix+'\t'.join([phenom.name(),str(sentence_count),sem_type,shell_key,\
                                              'Pr(shell|syn)',str(pr)])+'\n')
        except Exception: # probably because syn_to_type is the identity
            sem_types_enabled= False
            break

    """
    if not sem_types_enabled:
        for syn_key,shell_key in phenom.target_syn_shells():
            pr = math.exp(lexicon.sem_distribution._log_shell_given_type_prob(syn_key,shell_key,\
                                                                                  sem_store,sentence_count,'log'))
            f_out.write(prefix+'\t'.join([phenom.name(),str(sentence_count),syn_key,shell_key,\
                                              'Pr(shell|syn)',str(pr)])+'\n')
    """



def fork_and_run_dax(dax_filename,output_filename,phenom,lexicon,sentence_count,sem_store,RuleSet):
    """fork a process, train the model on a single example (given in filename), and then output the stats"""
    PID = os.fork()
    if PID == 0:
        f_dax = open(dax_filename)
        f_out_local = open(output_filename,'w')
        input_pairs = f_dax.readlines()
        f_dax.close()
        SemLearn.train_rules(sem_store,RuleSet,lexicon,True,input_pairs,[],\
                                 None,None,False,sentence_count,f_out_additional=f_out_local,\
                                 truncate_complex_exps=False)
        print_cat_stats(phenom,lexicon,sentence_count,sem_store,RuleSet,f_out_local,prefix='FILE'+dax_filename[-1]+' ')

        if dax_filename.endswith('10') or dax_filename.endswith('13') or dax_filename.endswith('16'):
            if dax_filename.endswith('10'):
                target_words = ['the', 'man', 'can', 'baby']
            elif dax_filename.endswith('13'):
                target_words = ['the', 'man', '\'ll', 'baby']
            elif dax_filename.endswith('16'):
                target_words = ['the', 'man', 'willx', 'baby']
            for target_word in target_words:
                print_syn_lf_given_word_probs('OtherWordsCheck',lexicon,target_word,\
                                                  sem_store,RuleSet,sentence_count,f_out_local)
            syntactic_categories = lexicon.catcounts.keys()
            for syn_key in syntactic_categories:
                print_syn_yield_distribution('SynYield',RuleSet,syn_key,sentence_count,f_out_local)

        f_out_local.close()
        print('Done.')
        os._exit(0)
    else:
        os.waitpid(PID,0)
        return


def main(output_fn, special_option='N', lexicon_fn = None, \
             lexicon = None, sentence_count = None, sem_store = None, RuleSet = None):

    if lexicon_fn:
        f_lexicon = open(lexicon_fn,'rb')
        lexicon,sentence_count,sem_store,RuleSet = pickle.load(f_lexicon)
        f_lexicon.close()

    dax_range = []

    if special_option == 'T':
        test_file = open("trainFiles/trainPairs_20","r")
        test_out = open(output_fn,"w")
        SemLearn.test(test_file,sem_store,RuleSet,lexicon,test_out,sentence_count)
        test_file.close()
        test_out.close()
    elif special_option in ['L', 'F', 'N']:
        synt_cat_distribution = get_synt_distribution(["((S\NP)/NP)", "((S/NP)/NP)", "((S\NP)\NP)", "((S/NP)\NP)"], lexicon, sem_store, RuleSet, sentence_count)
        D = get_transitive_cats(lexicon,sentence_count,sem_store,RuleSet,synt_cat_distribution)
        f_out = open(output_fn+'.trans_cats','w')
        for k,v in D.items():
            f_out.write(str(sentence_count)+'\t'+k+'\t'+str(v)+'\n')
        f_out.close()
        if special_option == 'L':
            return
        else:
            #sents_to_parse = [['this', 'is', 'corp', 'Mommy'], ['this', 'is', 'the', 'corp']]
            sents_to_parse = []
            #'Jarjar has it'.split(), 'what you did have'.split(), 'what did you have'.split()]
            #[('what you did have ?','lambda $0_{e}.lambda $1_{ev}.aux|do&PAST(v|have(pro|you,$0,$1),$1)'), 
            #('what did you have ?','lambda $0_{e}.lambda $1_{ev}.aux|do&PAST(v|have(pro|you,$0,$1),$1)')]

            if sents_to_parse != []:
                test_out = open(output_fn+'.parses','w')
                rubbish_f_out = open('rubbish','w')
                for sentnum,sent_to_parse in enumerate(sents_to_parse):
                    parser.parse(sent_to_parse,sem_store,RuleSet,lexicon,sentence_count,rubbish_f_out,\
                                     test_out_parses=test_out,target_top_cat='Swh')
                rubbish_f_out.close()
                test_out.close()

            f_out = open(output_fn,'w')
            for phenom in get_phenomena():
                print_cat_stats(phenom,lexicon,sentence_count,sem_store,RuleSet,f_out)

            # output the negation phenomena
            negation_target_words = ['not', '\'t', 'gonna']
            for target_word in negation_target_words:
                print_syn_lf_given_word_probs('Negation',lexicon,target_word,sem_store,RuleSet,sentence_count,f_out)

            # output the adverbs phenomena
            for target_word in ADVERBS:
                print_syn_lf_given_word_probs('Adverbs',lexicon,target_word,sem_store,RuleSet,sentence_count,f_out)

            # output the auxilliaries Pr(m|w)
            for target_word in AUXILIARIES:
                print_syn_lf_given_word_probs('Auxiliaries',lexicon,target_word,sem_store,RuleSet,sentence_count,f_out)

            # output the wh Pr(m|w)
            for target_word in WH_WORDS:
                print_syn_lf_given_word_probs('WH words',lexicon,target_word,sem_store,RuleSet,sentence_count,f_out)

            # output the prepositions
            preposition_target_words = ['on', 'in', 'with', 'good', 'big', 'nice', 'busy', 'right']
            for target_word in preposition_target_words:
                print_syn_lf_given_word_probs('PrepCheck',lexicon,target_word,sem_store,RuleSet,sentence_count,f_out)

            # output the yield distribution of some syntactic categories
            syntactic_categories = ['PP','NP','(PP/N)','(S\NP)','(S/NP)','((S\NP)/N)']
            for syn_key in syntactic_categories:
                print_syn_yield_distribution('SynYield',RuleSet,syn_key,sentence_count,f_out)
            dax_range = range(1,16)

    elif special_option != 'N':
        dax_range = [int(x) for x in special_option.split(':')]
        print(dax_range)
    
    if special_option != 'N':
        for ind in dax_range:
            if ind in [1,2]:
                phenom = get_dax_phenom()
            elif ind == 3:
                phenom = get_acorp_phenom()
            elif ind == 4:
                phenom = get_corp_phenom()
            else:
                continue
            fork_and_run_dax(PATH+'dax_examples/dax_example'+str(ind),\
                                 output_fn+'.out'+str(ind),phenom, \
                                 lexicon,sentence_count,sem_store,RuleSet)



if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: extract_from_lexicon.py <pickled file> <output> ' + 
              ' <special options: ' + \
                  'T for just test, number for a single dax, F otherwise]>')
        sys.exit(-1)
    main(sys.argv[2],sys.argv[3],sys.argv[1])
    print('Done.')




