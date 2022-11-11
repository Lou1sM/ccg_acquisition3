import extract_from_lexicon3
import pdb
import math, re

transitive_lf = re.compile(re.escape('lambda $0_{e}.lambda $1_{e}.lambda $2_{ev}.v|')+\
                   "[^()]+\\(\\$[01],\\$[01],\\$2\\)")
transitive_lf2 = re.compile(re.escape('lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.v|')+\
                   "[^()]+\\(\\$[01],\\$[01],\\$2\\)")

class VerbRepository:

    def __init__(self):
        word_lfs = extract_from_lexicon3.get_transitive_lfs()
        self._phenom = extract_from_lexicon3.Phenomenon('Transitives', word_lfs, '((S\NP)/NP)', \
                               ["((S\NP)/NP)", "((S/NP)/NP)", "((S\NP)\NP)", "((S/NP)\NP)"], \
                               two_LFs = True, \
                               flipped_target_syn = "((S/NP)\NP)",sem_type='((S|NP)|NP)',\
                               target_shells=['lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.placeholderP($0,$1,$2)',\
                                                  'lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.placeholderP($1,$0,$2)'])
        self.verb_instances = []
    
    def phenom(self):
        return self._phenom

    def update(self,other):
        for instance in other.verb_instances:
            if not instance in self.verb_instances:
                self.verb_instances.append(instance)

    def add_verb(self,semstring,lexicon,sem_store,RuleSet,sentence_count):
        """cat is an instance of cat.cat. checks if cat is a transitive and adds it."""
        try:
            if transitive_lf.match(semstring) or transitive_lf2.match(semstring):
                target_word = self._phenom.target_word(semstring)
            else:
                return
        except Exception:
            return
        pr_correct_w_given_lf = math.exp(lexicon.get_map_log_word_prob(target_word,\
                                          self.phenom().sem_type(),semstring,sentence_count))
        pr_correct_lf_given_w = extract_from_lexicon3.get_lf_given_word_probs(lexicon,\
                                          target_word,sem_store,RuleSet,sentence_count)
        pr_syn_given_lf = extract_from_lexicon3.get_synt_distribution(\
            ["((S\NP)/NP)", "((S/NP)/NP)", "((S\NP)\NP)", "((S/NP)\NP)"], \
                lexicon, sem_store, RuleSet, sentence_count)
        
        marginal_lf_probs = {}
        marginal_lf_probs[semstring] = \
            math.exp(lexicon.get_map_log_sem_prob('((S|NP)|NP)',semstring,sem_store))
        pr_syn = 0.0
        for syn_key in ["((S/NP)/NP)", "((S\NP)\NP)"]:
            pr_syn += pr_syn_given_lf.get(syn_key,0.0)
        other_lf = extract_from_lexicon3.swap_args(semstring)
        pr_other_lf = lexicon.get_map_log_sem_prob('((S|NP)|NP)',other_lf,sem_store)
        if pr_other_lf is None:
            marginal_lf_probs[other_lf] = 0.0
        else:
            marginal_lf_probs[other_lf] = math.exp(pr_other_lf) * pr_syn
        
        prob_syn_given_lf1_or_lf2 = \
            pr_syn_given_lf['((S\NP)/NP)'] * marginal_lf_probs.get(semstring) / \
            sum(marginal_lf_probs.values())
        
        w_given_LF = extract_from_lexicon3.get_w_dist_given_LF(lexicon,sentence_count,sem_store,RuleSet,semstring)
        w_given_other_LF = extract_from_lexicon3.get_w_dist_given_LF(lexicon,sentence_count,sem_store,RuleSet,other_lf)
        w_given_either_lf = (marginal_lf_probs[semstring] * w_given_LF.get(target_word,0.0) + \
                                 marginal_lf_probs[other_lf] * w_given_other_LF.get(target_word,0.0)) / \
                                 sum(marginal_lf_probs.values())
        
        self.verb_instances.append((sentence_count,target_word,semstring,\
                                        pr_correct_lf_given_w,\
                                        pr_correct_lf_given_w.get(semstring,0.0),\
                                        pr_correct_w_given_lf,\
                                        pr_syn_given_lf,\
                                        marginal_lf_probs,\
                                        w_given_other_LF,\
                                        w_given_either_lf,\
                                        prob_syn_given_lf1_or_lf2))


    def verb_prob_instances(self):
        """
        returns a list of 
        (verb,#occurrance,first occurance,total number of occurrances,Pr(syn,w|LF1 or LF2))
        """
        output = []
        freqs = {}
        for instance in self.verb_instances:
            freqs[instance[1]] = freqs.get(instance[1],0) + 1
        num_occurrance = {}
        first_occurrance = {}
        for instance in self.verb_instances:
            num_occurrance[instance[1]] = num_occurrance.get(instance[1],0) + 1
            if not first_occurrance.has_key(instance[1]):
                first_occurrance[instance[1]] = instance[0]
            pr_syn_w_given_lf = instance[10] * instance[9]
            output.append((instance[1],num_occurrance[instance[1]],first_occurrance[instance[1]],\
                              freqs[instance[1]],pr_syn_w_given_lf))
        return output


"""
OBSELETE PREV VERSION 14/5/15
    def verb_prob_instances(self):
        returns a list of 
        (verb,#occurrance,first occurance,total number of occurrances,Pr(syn,w|LF1 or LF2))
        output = []
        freqs = {}
        for instance in self.verb_instances:
            freqs[instance[1]] = freqs.get(instance[1],0) + 1
        num_occurrance = {}
        first_occurrance = {}
        for instance in self.verb_instances:
            num_occurrance[instance[1]] = num_occurrance.get(instance[1],0) + 1
            if not first_occurrance.has_key(instance[1]):
                first_occurrance[instance[1]] = instance[0]
            output.append((instance[1],num_occurrance[instance[1]],first_occurrance[instance[1]],\
                              freqs[instance[1]],instance[5]*instance[6]['((S\\NP)/NP)']))
        return output
"""
