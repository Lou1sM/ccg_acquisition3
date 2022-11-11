###########################################
# CLASSES USED TO CREATE THE LEXICON      #
###########################################

import re, pdb
import copy
import random
import math
from math import exp
from math import log
from tools import log_sum
from tools import inf
import scipy.special
from cat import synCat
from collections import defaultdict
from scipy.misc import logsumexp
from errorFunct import error

########################################
# lamda_op is to be used with the      #
# sem_rep below                        #
########################################
class lambda_op:
    def __init__(self,dep,separator):
        self.type = dep.type
        
        # only attach dep for use in building reps
        # as it lets us make sure that the lambda
        # reps point to the right place.
        # nullify before saving        
        self.dep = dep
        
        # separator is:
        #        None iff l refers to direct child
        #         dep iff l refers to child of dep
        #         lambda iff l refers to child of lambda
        self.separators = [separator]
    def add_separator(self,separator):
        self.separators.append(separator)
    def nullify(self):
        self.dep = None
        

########################################
# sem_rep is used to represent the l.f #
# and has functions to deal with       #
# decomposition                        #
########################################
class sem_rep:
    def __init__(self,ty,name,fineType):
        self.built = False
        self.type = ty
        self.fineType = fineType
        self.name = name
        self.dependents = []
        self.all_deps = []
        self.parents = []
        self.lambdas = []
        self.all_pairs = []
        self.sem_key = ''
        
    def set_key(self):
        self.sem_key = self.return_key(True)
    def build_rep(self,sem_c,sem_comps,comps):
        if not self.built:
            for d in sem_c.Dependents:
                self.add_dep(comps[d])
                dep = comps[d].build_rep(sem_comps[d],sem_comps,comps)
                for d2 in dep.all_deps:
                    if d2 not in self.all_deps:
                        self.all_deps.append(d2)
            self.set_prior()
            self.built = True
        return self
    def set_prior(self):
        # this maybe shouldn't be 0.5 (or should it??)
        #self.prior = pow(0.5,1+len(self.dependents))
        if len(self.dependents) == 0:
            self.prior = 1.0
        else:
            self.prior = 1.0
    def add_dep(self,dep):
        self.dependents.append(dep)
        self.all_deps.append(dep)
        dep.make_parent(self)
    def make_parent(self,parent):
        self.parents.append(parent)
    def copy(self):
        s = copy.deepcopy(self)
        return s
    def print_self(self):    
        for l in self.lambdas:
            pass 
            print 'lambda_'+l.type,'.',
        print self.name+'(',
        for l in self.lambdas:
            #if self.lambdas[l] == None:
            print l.type,',',
        for d in self.dependents:
            d.print_self()
            if self.dependents.index(d) != len(self.dependents)-1:
                print ',',
        print ')',
    def make_pairs(self,sem_store,lo):
        self.all_pairs = []
        i = 0
        for d in self.all_deps:
            s = self.copy()
            #s.all_pairs = []
            if s.all_deps[i].make_var():
                #    print >> lo,'made pairs for :: \\\\'
                #print >> lo,'$'+self.return_key(True)+'$'
                #print >> lo, '\\\\'
                #print  >> lo,'is :: \\\\'
                #print  >> lo,'$'+s.return_key(True)+'~~~~~~'+d.return_key(True)+'$'
                #print >> lo,'\\vspace{1cm} \\\\'
                s.nullify_lambdas()                
                
                self.all_pairs.append((s.return_key(True),d.return_key(True)))
                
                s.set_key()
                s.set_prior()
                
                if not sem_store.check(s.sem_key):
                    s.make_pairs(sem_store,lo)
                    sem_store.add(s)
                
                d.set_key()
                d.set_prior()
                if not sem_store.check(d.sem_key):
                    d.make_pairs(sem_store,lo)
                    sem_store.add(d)
            else:
            #    print 'got issues'
                # most of which chime nicely with those 
                # things that CCG also has issues with
            #    print self.return_key(True)
            #    print d.return_key(True)
                pass
            i+=1
        print >> lo,'made pairs for :: \\\\'
        print >> lo,'$'+self.return_key(True)+'$'
        #print >> lo, '\\\\'
        #print  >> lo,'are :: \\\\'
        for p in self.all_pairs:
            print  >> lo,'$'+p[0]+'~~~~~~'+p[1]+'$\\\\'
        #print >> lo,'\\vspace{1cm} \\\\'
    def check_parent_branch(self,parent):
        #################################
        ## this function checks whether #
        ## a given dependent has a parent
        ## on a different branch.
        ## this only returns true if the#
        ## node commands the piece of   #
        ## l.f. itself.                    #
        #################################
        
        if self.parents == []:
            return True
    
        pb = False
        for p in self.parents:
            if p != parent:
                if p.check_parent_branch(parent):
                    pb = True
        return pb
        
        
    def make_var(self):
        
        ####################################
        # This is to check if any of the 
        # dependents are called by another
        # branch of the semantic graph in 
        # which case we do not want to make
        # this a variable until they've been
        # dealt with.
        ####################################
        other_branch = False
        for d in self.all_deps:
            if d.check_parent_branch(self):
                other_branch = True
        ####################################
        
        if not other_branch:
            for p in self.parents:
                if p.make_lambda(self, None) is None:
                    return False
            self.parents = []
            return True
        else:
            #print self.all_deps
            #print self.return_key(True)
            #for d in self.all_deps:
                #print d.return_key(True)
            return False
            
    def make_lambda(self,dep,separator):
        if dep in self.all_deps:            
            ##########################################
            # separator is the !node! that separates #
            # the dependent from the parent (often   #
            # None).                                 #
            ##########################################
            del_deps = []
            
            ##########################################
            # need to also delete lambdas shared with#
            # the child lambda list                     #
            ##########################################

            self.lambdas.reverse()
            dep.lambdas.reverse()
                
            subs = False
            
            i = 0
            del_lambdas = []
            for l in dep.lambdas:
                
                if l.type == self.lambdas[i].type and (len(self.lambdas[i].separators) == 1): 
                    # check that the lambda is of the correct type and
                    # that it only points to one place before deleting
                    # it
                    del_lambdas.append(i)
                else:
                    return None
                    
                # Need to sort out substitution stuff here #
                # DON'T GIVE UP #
                    
                #elif l.type == self.lambdas[i].type:# and (len(dep.lambdas) == len(self.lambdas) ==1) and (len(self.lambdas[i].separators) == 2):
                    #del_seps = []
                    #j = 0
                    #for s in self.lambdas[i].separators:
                        #if s == dep:
                            #del_seps.append(j)
                    #del_seps.reverse()
                    #for s in del_seps:
                        #del self.lambdas[i].separators[s]
                    #if len(self.lambdas[i].separators) == 0:
                        #del_lambdas.append(i)
                    #subs = True
                    ## need to sort this out way better
                    ## is this happening when it shouldn't?
                    ## actually should be working out if the
                    ## term pointed at has or had two parents
                    ##return None
                #elif len(self.lambdas[i].separators) > 2:
                    #print 'shared between more than two'
                    #error()
                #else:
                
                    #return None
                    
                    # all the existing lambda
                    # terms for the dependent
                    # should have a counterpart
                    # at front up top. 
                    #        If not :: FAIL
                i+=1
        
            self.lambdas.reverse()
            dep.lambdas.reverse()
            
            orig_length = len(self.lambdas)
            for i in del_lambdas:
                del self.lambdas[orig_length-1-i]
                
                #else:
                    #for s in self.lambdas[orig_length-1-i].separators:
                        #if s == dep:
                            #print 'is equal'
                            #del_s.append(j)
                        #else:
                            #print 'not equal'
                #j += 1        

                #for s in del_s.reverse():
                    #del.self.lambdas[orig_length-1-i].separators        
                
                #if len

            self.lambdas.append(lambda_op(dep,separator))

            ###########################################
            # This is only for substitution type rules# 
            # and swaps the order of the (now 2) lamda#
            # terms.                                  #
            ###########################################
            if subs:
                if not len(self.lambdas)==2:
                    print 'not 2. this is mental'
                    print len(self.lambdas)
                    error("not 2. this is mental")
                self.lambdas.reverse()
            ###########################################
            
            ###########################################
            if dep in self.dependents:
                del self.dependents[self.dependents.index(dep)]
            del self.all_deps[self.all_deps.index(dep)]
            ###########################################
            
            
            ###########################################
            # Under_Deps are given the binary option  #
            # to go or stay                              #
            ###########################################
            for d in self.all_deps:
                if d in dep.all_deps:
                    #if d in self.dependents
                    del_deps.append(d)
                    
                    #if 
                    # but if it's in dependents (direct)
                    # too then it will need a lambda term
                    # _or_ it should stay here and the 
                    # other thing should get a lambda
            ##############################
            # should there be something  #
            # to deal with more complex  #
            # semantic possibilities?    #
            ##############################        
            for p in self.parents:
                pr = p.make_lambda(dep,self)
                if pr is None:
                    
                    return None
            # delete from all_deps here #
        
            for d in del_deps:
                del self.all_deps[self.all_deps.index(d)]
        #for d in self.dependents:
            #d.make_lambda(dep,
            return True
        else:
            double_dep = False
            for l in self.lambdas:
                if l.dep == dep:
                    l.add_separator(separator)
                    double_dep = True
            if not double_dep:
                error("")
            return True
    # why does doing it like this help ? #
    # buddy, I'm not sure that it does.  #
    
    # have just recreated what I had before
    # need to support lx.f(x(a)) which I 
    # don't
    
    # do I really need to support this?
    # KIND OF IS SUPPORTED. SUBJ STILL 
    # THERE
    
    # NEED TO SPLIT SHARED ITEMS WHEN 
    # NECCESSARY. WOULD BE WELL NICE 
    # IF THIS WAS OPTIONAL TOO 
        
        
        
        
    def combine(self,target):
        # THIS ONLY DOES SEMANTIC COMBINATION #
        if self.lambdas[-1].type == target.type:
            for s in self.lambdas[-1].separators:
                if s is None:
                    self.dependents.append(target)
                
                else:
                    s.combine(target)
            self.lambdas.pop()
            return True
        else:
            return False
            #self.dep
    
    def return_key(self,top):
        key = ''
        if top:
            self.lambdas.reverse()
            for l in self.lambdas:
                key = key+'\lambda_{'+l.type+'}'
            self.lambdas.reverse()
            
            if len(self.lambdas)!=0:
                key = key+'.'
        key = key+self.name+'_{'+self.type+'}'
        if self.dependents != [] or self.lambdas!= []:
            key = key+'('
        i = 0
        for d in self.dependents:
            key = key+d.return_key(False)
            if i < len(self.dependents)-1:
                key = key+','
            i += 1
        
        #key = key+')'
        lambda_done = False
        for l in self.lambdas:
            if None in l.separators:
                lambda_done = True
                key = key+','+l.type
            
        #if top:
        if self.dependents != [] or lambda_done:
            key = key+')'
        return key
    def nullify_lambdas(self):
        for l in self.lambdas:
            l.nullify()

########################################
# this syn_cat is used to generate all #
# syntactic parses                     #
########################################
class syn_cat:
    def __init__(self,head,targets):
        self.head = head
        self.targets = targets
        self.key = self.return_key()
        # [(cat,direction),....,(cat,direction)]
    def add_target(self,cat,direction):
        self.targets.append((cat,direction))
        self.key = self.return_key()
    def extend_targets(self,targets):
        self.targets.extend(targets)
        self.key = self.return_key()
    def remove_targets(self,n):
        togo = self.targets[-n:]
        del self.targets[-n:]
        self.key = self.return_key()
        return togo
    def copy(self):
        s = copy.deepcopy(self)
        return s
    def return_key(self):
        key = self.head
        for t in self.targets:
            if t[1] == 'fwd':
                key = key+'/'+t[0]
            elif t[1] == 'back':
                key = key+'\\'+t[0]
            else:
                'not fwd or back - WHAAAT?'
                error('not fwd or back - WHAAAT?')
        return key

########################################
# Function to build rep from sem_comps #
# that are read in from the input       #
########################################
def make_reps(sem_c,sem_comps,Parent,targ_rep,sem_store,lo):
    comps = []
    for c in sem_comps:
        comps.append(sem_rep(c.Type,c.Name,c.fineType))
    rep = comps[sem_comps.index(sem_c)].build_rep(sem_c,sem_comps,comps)
    #rep.print_self()
    rep.make_pairs(sem_store,lo)
    rep.set_key()
    sem_store.add(rep)
    return rep.sem_key
########################################
    



########################################
class lexical_item:
    def __init__(self,word,syn_key,sem_key,is_shell_item=False):
        self.last_used = 0
        self.word = word
        self.sem_key = sem_key
        self.syn = syn_key
        self.key = (self.word,self.sem_key,self.syn)
        #self.sem_syn = {}
        self.alpha = 0
        self.bleurgh_alpha = 0.0
        self.top_term = 0
        self.prior = 0
        self.num_seen = 0
        self.is_shell_item = is_shell_item

    def set_word_prior(self):
        no_char = len(self.word) - self.word.count(' ')
        no_bound = self.word_count(' ')        
        w_prior = pow((float(499)/(26*500)),no_char)*pow(float(1/500),no_bound)
        return w_prior
    def increment_num_seen(self):
        self.num_seen += 1

    def update_alpha(self,alpha):
        #print "alpha was ",self.alpha
        #print "update is ",alpha
        self.alpha = self.alpha+alpha
        if self.alpha<-1E-5:
            print "negative alpha for ",self.toString()
            print "alpha is ",self.alpha
            error("alpha is negative")
            
    def update_bleurgh_alpha(self,alpha):
        self.bleurgh_alpha = self.bleurgh_alpha+alpha
    def set_bleurgh_alpha(self):
        self.bleurgh_alpha = self.alpha
    def print_alpha(self,out):
        print >> out,'<',self.word,',',self.sem,',',self.syn,'>   ::  a = ',self.alpha
        
    def update_p(self,prob):
        self.top_term += prob
    def set_last_used(self,sentence_count):
            self.last_used = sentence_count
    def clear_probs(self):
        self.top_term = 0    
    def toString(self):
        return self.word+" : "+ self.syn+" : "+self.sem_key
########################################
        
class sem_to_word:
    def __init__(self,sem_key,word):
        self.sem_key = sem_key
        self.word = word
        self.alpha = 0.0
        self.lex_items = []
        self.last_used = 0
    def add_lex_item(self,l):
        self.lex_items.append(l)
    def del_lex_item(self,l):
        del self.lex_items[self.lex_items.index(l)]
    def increment_alpha(self,update):
        self.alpha += update
    def set_last_used(self,count):
        self.last_used = count
        #sw.set_last_used(max(sw.last_used

########################################

typing_regexp = re.compile("[\\\\/]")
syn_to_type = lambda syn_key: (typing_regexp.sub("|",syn_key) if syn_key is not None else None)

class sem_distribution:

    def __init__(self, alpha_shell, alpha_shell_to_sem):
        
        self.type_to_count = {}
        self.type_shell_to_count = {}
        self.type_shell_sem_to_count = {}
        self.sem_to_pairs = defaultdict(list) # maps a sem_key to all the pairs (type,shell) with which it appears
        self.alpha_shell = alpha_shell # the alpha_0 of the type->shell distribution
        self.alpha_shell_to_sem = alpha_shell_to_sem # the alpha_0 of the shell->sem distribution
        
        self.use_special_alpha_o = False
        self.special_alpha_o = 100 * alpha_shell
        
        self.ignore_shells = False # generates the sem_key directly from the type
        self.alpha_type_to_sem = alpha_shell # the alpha_0 of the type->sem distribution; only used when ignore_shells is True
        
        #self.total_alpha = 0.0
        #self.sem_to_count = {}
        #self.sem_to_last_used = {}
        #self.type_sem_to_last_used = {}
        
    """
    def del_entry(self,syn_key,shell_sem_key,sem_key):
        "deletes the relevant entry from the distribution"
        sem_type = self.syn_to_type(syn_key)
        if self.has_key(syn_key,shell_sem_key,sem_key):
            cur_alpha = self.type_shell_sem_to_count[(sem_type,shell_sem_key,sem_key)]
            self._update(self.type_to_count,sem_type,-cur_alpha)
            self._update(self.type_shell_to_count,(sem_type,shell_sem_key),-cur_alpha)
            self._update(self.type_shell_sem_to_count,(sem_type,shell_sem_key,sem_key),-cur_alpha)
            #del self.type_shell_sem_to_count[(sem_type,shell_sem_key,sem_key)]
    """
    
    def check(self,syn_key,shell_sem_key,sem_key):
        sem_type = syn_to_type(syn_key)
        if not self.type_shell_to_count.has_key((sem_type,shell_sem_key,sem_key)):
            self.type_shell_sem_to_count[(sem_type,shell_sem_key,sem_key)] = 0.0
            self.sem_to_pairs[sem_key].append((sem_type,shell_sem_key))
            if not self.type_shell_to_count.has_key((sem_type,shell_sem_key)):
                self.type_shell_to_count[(sem_type,shell_sem_key)] = 0.0
                if not self.type_to_count.has_key(sem_type):
                    self.type_to_count[sem_type] = 0.0

    def get_most_common_shell(self,syn_key):
        sem_type = syn_to_type(syn_key)
        m = max([(k[1],v) for k,v in self.type_shell_to_count.items() if k[0] == sem_type],key=lambda x: x[1])
        return m[0]

    def all_sems_for_shell(self,syn_key,shell_key):
        sem_type = syn_to_type(syn_key)
        return [x[2] for x in self.type_shell_sem_to_count.keys() if x[0] == sem_type and x[1] == shell_key]

    def types_for_sem(self,sem_key):
        """
        Returns all the types for the given sem key.
        """
        all_pairs = self.sem_to_pairs[sem_key]
        return [x[0] for x in all_pairs]
                                            
    def all_shells_alphas(self,syn_key):
        """
        Returns all the shell semantic forms for a given syn_key and their alphas.
        """
        sem_type = syn_to_type(syn_key)
        return [(k[1],v) for k,v in self.type_shell_to_count.items() if k[0]==syn_key]

    def has_key_sem(self,sem_key):
        return self.sem_to_pairs.has_key(sem_key)

    def has_key(self,syn_key,shell_sem,sem_key):
        sem_type = syn_to_type(syn_key)
        return self.type_shell_sem_to_count.has_key((sem_type,shell_sem,sem_key))
    
    def has_key_shell(self,syn_key,shell_sem):
        sem_type = syn_to_type(syn_key)
        return self.type_shell_to_count.has_key((sem_type,shell_sem))

    def log_sem_prob(self,syn_key,shell_sem_key,sem_key,sem_store,sentence_count):
        """
        returns the probability Pr(sem_key|type), where type is determined by syn_key
        """
        if self.ignore_shells:
            return self._log_sem_given_type_prob(syn_key,shell_sem_key,sem_key,sem_store,sentence_count,'psi')
        else:
            return self._log_sem_given_shell_prob(syn_key,shell_sem_key,sem_key,sem_store,sentence_count,'psi') + \
                self._log_shell_given_type_prob(syn_key,shell_sem_key,sem_store,sentence_count,'psi')

    def map_log_sem_prob(self,syn_key,shell_sem_key,sem_key,sem_store,sentence_count):
        if self.ignore_shells:
            return self._log_sem_given_type_prob(syn_key,shell_sem_key,sem_key,sem_store,sentence_count,'log')
        else:
            return self._log_sem_given_shell_prob(syn_key,shell_sem_key,sem_key,sem_store,sentence_count,'log') + \
                self._log_shell_given_type_prob(syn_key,shell_sem_key,sem_store,sentence_count,'log')

    def _log_sem_given_shell_prob(self,syn_key,shell_sem_key,sem_key,sem_store,sentence_count,psi_or_log):
        """
        returns the probability Pr(sem_key|shell,type) where type is determined by syn_key.
        if psi_or_log equals 'log', it returns the MAP estimator, otherwise it returns the 
        estimator using psi (i.e., with the log-parametrization).
        """
        sem_type = syn_to_type(syn_key)
        log_prior = sem_store.get_log_prior(sem_key)
        scale = 1.0
        func = (scipy.special.psi if psi_or_log == 'psi' else math.log)
        log_prob = log_prior + func(scale*self.alpha_shell_to_sem) - \
            func(scale*self.type_shell_to_count[(sem_type,shell_sem_key)]  + scale*self.alpha_shell_to_sem)
        seen_log_prob = -inf
        if self.has_key(syn_key,shell_sem_key,sem_key):
            cur_alpha = self.alpha_type_shell_sem(sem_type,shell_sem_key,sem_key)
            if cur_alpha > 0:
                seen_log_prob = func(scale*cur_alpha) - \
                    func(scale*self.type_shell_to_count[(sem_type,shell_sem_key)]  + scale*self.alpha_shell_to_sem)
        if seen_log_prob > 0:
            seen_log_prob = 0.0
        log_prob = log_sum(log_prob, seen_log_prob)
        return log_prob

    def _log_shell_given_type_prob(self,syn_key,shell_sem_key,sem_store,sentence_count,psi_or_log):
        """returns Pr(shell|type), where type is determined by the syntactic type"""
        sem_type = syn_to_type(syn_key)
        log_prior = sem_store.get_log_prior(shell_sem_key)
        scale = 1.0
        func = (scipy.special.psi if psi_or_log == 'psi' else math.log)
        log_prob = log_prior + func(scale*self.alpha_shell) - \
            func(scale*self.type_to_count[sem_type] + scale*self.alpha_shell)
        seen_log_prob = -inf
        if self.has_key_shell(syn_key,shell_sem_key):
            cur_alpha = self.alpha_type_shell(sem_type,shell_sem_key)
            if cur_alpha > 0:
                seen_log_prob = func(scale*cur_alpha) - \
                    func(scale*self.type_to_count[sem_type] + scale*self.alpha_shell)
        if seen_log_prob > 0:
            seen_log_prob = 0.0
        log_prob = log_sum(log_prob, seen_log_prob)
        return log_prob

    def _log_sem_given_type_prob(self,syn_key,shell_sem_key,sem_key,sem_store,sentence_count,psi_or_log):
        """
        returns Pr(sem|type), where type is determined by the syntactic type
        """
        sem_type = syn_to_type(syn_key)
        if self.use_special_alpha_o: # and sem_type == "((S|NP)|NP)":
            alpha_o = self.special_alpha_o
        else:
            alpha_o = self.alpha_type_to_sem
        
        log_prior = sem_store.get_log_prior(sem_key)
        scale = 1.0
        func = (scipy.special.psi if psi_or_log == 'psi' else math.log)
        log_prob = log_prior + func(scale*alpha_o) - \
            func(scale*self.type_to_count[sem_type]  + scale*alpha_o)
        seen_log_prob = -inf
        if self.has_key(syn_key,shell_sem_key,sem_key):
            cur_alpha = self.alpha_type_shell_sem(sem_type,shell_sem_key,sem_key)
            if cur_alpha > 0:
                seen_log_prob = func(scale*cur_alpha) - \
                    func(scale*self.type_to_count[sem_type]  + scale*alpha_o)
        if seen_log_prob > 0:
            seen_log_prob = 0.0
        log_prob = log_sum(log_prob, seen_log_prob)
        return log_prob
        

    """
    def _multiply_param(self,D,last_used,key,learning_rates,sentence_count):
        factor = 1.0
        for i in range(last_used,sentence_count):
            factor = factor * (1 - learning_rates[i])
        D[key] = D[key] * factor
        return max(last_used,sentence_count)
    
    def refresh_sem_params(self,syn_key,shell_sem_key,sem_key,learning_rates,sentence_count):
        sem_type = self.syn_to_type(syn_key)
        if self.type_shell_sem_to_count.has_key((sem_type,shell_sem_key,sem_key)):
            self.lu_type_shell_sem_to_count[(sem_type,shell_sem_key,sem_key)] = \
                self._multiply_param(self.type_shell_sem_to_count,\
                                                     self.lu_type_shell_sem_to_count[(sem_type,shell_sem_key,sem_key)],\
                                                     (sem_type,shell_sem_key,sem_key),learning_rates,sentence_count)
        if self.type_shell_to_count.has_key((sem_type,shell_sem_key)):
            self.lu_type_shell_to_count[(sem_type,shell_sem_key)] = \
                self._multiply_param(self.type_shell_to_count,\
                                                     self.lu_type_shell_to_count[(sem_type,shell_sem_key)],(sem_type,shell_sem_key),\
                                                     learning_rates,sentence_count)
        if self.type_to_count.has_key(sem_type):
            self.lu_type_to_count[sem_type] = \
                self._multiply_param(self.type_to_count,self.lu_type_to_count[sem_type],sem_type,learning_rates,sentence_count)
    """

    def refresh(self,learning_rate):
        for my_dict in [self.type_to_count,self.type_shell_to_count,self.type_shell_sem_to_count]:
            my_dict.update((x,y*(1-learning_rate)) for x,y in my_dict.items())
    
    def alpha_type_shell_sem(self,sem_type,shell_sem_key,sem_key):
        return self.type_shell_sem_to_count[(sem_type,shell_sem_key,sem_key)]

    def alpha_type_shell(self,sem_type,shell_sem_key):
        return self.type_shell_to_count[(sem_type,shell_sem_key)]

    def sum_alphas(self,sem_key):
        """the sum of all alphas for this sem_key"""
        return sum([self.type_shell_sem_to_count[(type_shell[0],type_shell[1],sem_key)] \
                        for type_shell in self.sem_to_pairs[sem_key]])

    def _update(self,D,k,v):
        new_val = D[k] + v
        if new_val > 0:
            D[k] = new_val

    def update_alpha(self,syn_key,sem_shell,sem_key,val):
        sem_type = syn_to_type(syn_key)
        self.type_to_count[sem_type] = max(self.type_to_count[sem_type] + val, 0.0)
        self.type_shell_to_count[(sem_type,sem_shell)] = max(self.type_shell_to_count[(sem_type,sem_shell)] + val, 0.0)
        self.type_shell_sem_to_count[(sem_type,sem_shell,sem_key)] = max(self.type_shell_sem_to_count[(sem_type,sem_shell,sem_key)] + val, 0.0)

    #def all_alphas(self):
    #    return self.sem_to_count.items()
    
    #def alpha(self,sem_key):
    #    return self.sem_to_count[sem_key]
    


########################################
class Lexicon:
    mwe = True
    extrascale = 1.0
    
    def __init__(self,type_to_shell_alpha_o,shell_to_sem_alpha_o,word_alpha_o):
        self.word_alpha_o = word_alpha_o
        self.sentence_count = 0
        self.learningrates = []
        self.updateweight = 0.1
        self.wordprobfile = None
        self.wordstocheck = None
        self.fastMapFile = None
        self.mwe = Lexicon.mwe
        self.cur_cats = []
        
        self.lex = {}  # a mapping of (word,syn_key,sem_key) to lexical item
        self.words = {} # a mapping of words and MWEs to a list of triplets (word,syntax,semantcs) in which they occur
        self.sem_to_word = {} # maps a lambda expression and a word/MWE into a sem_to_word, which keeps the distribution of Pr(word|sem)
        
        self.semtoshell = {} # a mapping from a sem_key to a shell logical form
        self.catcounts = {} # exactly the same as lexicon.syntax[syn_key].alpha_top
        self.sem_distribution = sem_distribution(type_to_shell_alpha_o,shell_to_sem_alpha_o)
        self.updates = {} # a temporary lexicon in which updates to be made in the M step are saved
        
        # COMMENTED OUT BY OMRI
        #        self.categories = {} # ???
        #self.catToRepShell = {} # maps a syntactic category to a dict mapping logical forms to counts
        #self.wsems = {} # ???
        #self.repShells = {}
        #self.semantics = {} # maps a logical form to a number
        #self.syn_sem = {} # maps a (syn_key,sem_key) pair into a syn_sem instance
        #self.syn_top = syn_top
        #self.orig_syn_top = syn_top
        #self.syn_sem_top = syn_sem_top
        #self.orig_syn_sem_top = syn_sem_top
        #syn_sem.syn_sem_top = syn_sem_top
        #self.all_cur_lfs = set() # temporary storage
        #self.sem_alpha_o = sem_alpha_o
        
    def set_words_to_check(self,wordstocheck):
        self.wordstocheck = wordstocheck

    def get_log_shell_given_type_prob(self,syn_key,shell_sem_key,sem_store,sentence_count):
        return self.sem_distribution._log_shell_given_type_prob(syn_key,\
                          shell_sem_key,sem_store,sentence_count,'log')
        
    def get_map_word_given_shell(self,word,syn_key,shell_key,sentence_count,sem_store):
        log_probs = []
        # adding the new sem prob
        log_probs.append(Lexicon.log_word_prior(word) + \
            self.sem_distribution._log_shell_given_type_prob(syn_key,shell_key,sem_store,sentence_count,'log'))
        for old_sem in self.sem_distribution.all_sems_for_shell(syn_key,shell_key):
            log_probs.append(self.get_map_log_word_prob(word,syn_key,old_sem,sentence_count,psi_or_log='log') + \
                                 self.get_map_log_sem_prob(syn_key,old_sem,sem_store))
        return scipy.misc.logsumexp(log_probs)
    
    def check(self,word,syn_key,sem_key,sem):
        """
        OMRI:
        adds the triplet (word,syn_key,sem_key) into the lexicon.
        sem is an instance of exp.exp which has the key sem_key (I think).
        """
        if "placeholderW" in word:
            return
        key = (word,syn_key,sem_key)
        
        if not self.lex.has_key(key):
            li = lexical_item(word,syn_key,sem_key)
            self.lex[key] = li
        else:
            li = self.lex[key]
            return
        
        if not self.sem_to_word.has_key((sem_key,word)):
            sw = sem_to_word(sem_key,word)
            sw.add_lex_item(li)
            self.sem_to_word[(sem_key,word)] = sw
        else:
            sw = self.sem_to_word[(sem_key,word)]
            sw.add_lex_item(li)
        
        # if a new syntax has appeared
        if not self.catcounts.has_key(syn_key):
            self.catcounts[syn_key] = 0.0
        
        if sem is not None:
            shell_sem_key = sem.toStringShell(True)
        else:
            shell_sem_key = sem_key
        
        if not self.sem_distribution.has_key(syn_key,shell_sem_key,sem_key):
            self.sem_distribution.check(syn_key,shell_sem_key,sem_key)
        
        if not self.semtoshell.has_key(sem_key):
            self.semtoshell[sem_key] = shell_sem_key
        
        if not self.words.has_key(word):
            self.words[word] = [key]
        else:
            self.words[word].append(key)

    """
    def getWordsFromSynSem(self,ccgcat,maxw):
        ssk = (ccgcat.syn.toString(),ccgcat.sem.toString(True))
        if not self.syn_sem.has_key(ssk): return []
        #print "for ",ccgcat.toString()," got ",self.syn_sem[(ccgcat.syn.toString(),ccgcat.sem.toString(True))]
        words = []
        for l in self.syn_sem[ssk].words.values():
            words.append((l.alpha,l.word))
        words.sort()
        words.reverse()
        wr = []
        for i in range(min(len(words),maxw)):
            print "w1 is ",words[i][1].split()
            wr.append(words[i][1].split())
        return wr
    """
    
    def get_lex_items(self,word,test_out=None,guess=False,sem_store=None,beamsize=None):
        """
        Returns the most common lexical items with this specific word.
        """
        pi = []
        if beamsize is None:
            beamsize = 20
        if self.words.has_key(word):
            pi = [self.lex[item] for item in self.words[word]]
            pi.sort(key=lambda x: -x.alpha)
            pi = pi[:20]
        elif guess:
            candlist = []
            common_cats = sorted(self.catcounts.items(),key=lambda x:-x[1])[:beamsize]
            for c in common_cats:
                synkey = c[0]
                shell_key = self.sem_distribution.get_most_common_shell(synkey)
                lex_item = lexical_item(word,synkey,shell_key,True)
                pi.append(lex_item)
        for l in pi:
            if test_out: print >> test_out, l.toString()
        return pi

    """
    def get_common_cats(self):
        catlist = []
        for syn_key,count in self.catcounts.items():
            catlist.append((count,syn_key))
        catlist.sort()
        catlist.reverse()
        return catlist[:20]
    """

    def get_map_log_word_prob(self,word,syn_key,sem_key,sentence_count,psi_or_log='log'):
        if word == "placeholderW": return -inf
        if "placeholder" in sem_key: return Lexicon.log_word_prior(word)
        self.refresh_params(word,syn_key,sem_key,self.sentence_count-1)
        if self.sem_to_word.has_key((sem_key,word)):
            self.refresh_sem_word_params(sem_key,word)
        shell_sem_key = self.semtoshell.get(sem_key,None)
        if shell_sem_key is None:
            print('Sem key error:'+sem_key)
            return 0.0
        sem_type = syn_to_type(syn_key)
        
        func = (math.log if psi_or_log == 'log' else scipy.special.psi)
        scale = 1.0
        if self.sem_distribution.has_key(syn_key,shell_sem_key,sem_key):
            if self.sem_to_word.has_key((sem_key,word)):
            
                sum_alpha = self.sem_distribution.alpha_type_shell_sem(sem_type,shell_sem_key,sem_key)
                alpha = self.sem_to_word[(sem_key,word)].alpha
            
                log_prior = Lexicon.log_word_prior(word) + \
                    func(scale*self.word_alpha_o) - func(scale*sum_alpha + scale*self.word_alpha_o)
            
                if word.find(" ")!=-1 and not self.mwe: log_prior = -inf
                seen_log_prob = -inf
                if alpha > 0:
                    seen_log_prob = func(scale*alpha) - func(sum_alpha + scale*self.word_alpha_o)
                if seen_log_prob > 0:
                    seen_log_prob = 0.0
                log_prob = log_sum(log_prior,seen_log_prob)

                return log_prob
            else:
                sum_alpha = self.sem_distribution.alpha_type_shell_sem(sem_type,shell_sem_key,sem_key)
                log_prior = Lexicon.log_word_prior(word) + func(scale*self.word_alpha_o) - \
                    func(scale*sum_alpha + scale*self.word_alpha_o)
                return log_prior
        else:
            error("sem distribution doesn't have the required key")
    
    def get_log_word_prob(self,word,syn_key,sem_key,sentence_count):
        """
        OMRI: This is the method actually used in inside_outside_calc for computing the chart.
        """
        return self.get_map_log_word_prob(word,syn_key,sem_key,sentence_count,psi_or_log='psi')
    
    def get_map_log_sem_prob(self,syn_key,sem_key,sem_store):
        """
        Returns the log of the MAP estimate for the semantic key sem_key.
        If syn_key == None, then the probability is marginalized over the syntactic types.
        """
        shell_sem_key = self.semtoshell.get(sem_key,None)
        if shell_sem_key is None:
            print('Sem key error:'+sem_key)
            return None
        return self.sem_distribution.map_log_sem_prob(syn_key,shell_sem_key,sem_key,sem_store,self.sentence_count)
        
    def get_log_sem_prob(self,syn_key,sem_key,sem_store):
        shell_sem_key = self.semtoshell.get(sem_key,None)
        if shell_sem_key is None:
            print('Sem key error:'+sem_key)
            return None
        return self.sem_distribution.log_sem_prob(syn_key,shell_sem_key,sem_key,sem_store,self.sentence_count)
    
    """
    def get_sem_prob(self,syn_key,sem_key,sem_store):
        shell_sem_key = self.semtoshell[sem_key]
        return math.exp(self.sem_distribution.log_sem_prob(syn_key,shell_sem_key,sem_key,sem_store,self.sentence_count))
    """
    
    def update_params(self,word,syn_key,sem_key,prob,sentence_count):
        self.lex[(word,syn_key,sem_key)].update_alpha(prob)
        sem_shell = self.semtoshell[sem_key]
        self.sem_distribution.update_alpha(syn_key,sem_shell,sem_key,prob)
        self.sem_to_word[(sem_key,word)].increment_alpha(prob)
        self.catcounts[syn_key] += prob
        #self.catToRepShell[syn_key][self.semtoshell[sem_key]] += prob
        #self.syn_sem[(syn_key,sem_key)].update_alpha(prob)
        #self.syntax[syn_key].update_alpha_top(prob)
    
    def refresh_all_params(self,sentence_count):
        for l in self.lex:
            self.refresh_params(l[0],l[1],l[2],sentence_count)
    
    """
    def refresh_sem_params(self,syn_key,sem_key):
        sem_shell = self.semtoshell[sem_key]        
        if self.sem_distribution.has_key(sem_key):
            for i in range(self.sem_distribution.last_used(syn_key,sem_key),self.sentence_count):
                learning_rate = self.learningrates[i]
                update = -learning_rate*self.sem_distribution.alpha(sem_key)
                self.sem_distribution.update_alpha(syn_key,sem_shell,sem_key,update)
                #if sem_key.find("Placeholder")!=-1:
                #    self.catToRepShell[syn_key][self.semtoshell[sem_key]] + update
            max_iter = max(self.sem_distribution.last_used(syn_key,sem_key),self.sentence_count)
            self.sem_distribution.set_last_used(syn_key,sem_key,max_iter)
    """
            
    def refresh_sem_word_params(self,sem_key,word):
        if self.sem_to_word.has_key((sem_key,word)):
            sw = self.sem_to_word[(sem_key,word)]
            for i in range(sw.last_used,self.sentence_count):
                learning_rate = self.learningrates[i]
                update = -learning_rate*sw.alpha
                sw.increment_alpha(update)
            sw.set_last_used(max(sw.last_used,self.sentence_count))
            
    def refresh_params(self,word_key,syn_key,sem_key,sentence_count):
        if self.lex.has_key((word_key,syn_key,sem_key)):
            l = self.lex[(word_key,syn_key,sem_key)]
            for i in range(l.last_used,self.sentence_count):
                learning_rate = self.learningrates[i]
                l.update_alpha(-learning_rate*l.alpha)
            l.set_last_used(max(l.last_used,self.sentence_count))

        if self.sem_to_word.has_key((sem_key,word_key)):
            self.refresh_sem_word_params(sem_key,word_key)
        
        #self.sem_distribution.refresh_sem_params(syn_key,self.semtoshell[sem_key],sem_key,\
        #                                             self.learningrates,self.sentence_count)
        

    def store_log_update(self,word,syn_key,sem_key,log_prob):
        prob = exp(log_prob)
        if not self.updates.has_key((word,syn_key,sem_key)):
            self.updates[(word,syn_key,sem_key)] = prob
        else: self.updates[(word,syn_key,sem_key)] += prob
    
    def get_learning_rate(self,i):
        return self.learningrates[i]

    def set_learning_rates2(self,datasize,k):
        self.learningrates = []
        To = 50 
        k = -0.8
        print('K k='+str(k))
        for i in range(1,datasize+1):
            self.learningrates.append(pow(To+i,k))
    
    def set_learning_rates(self,datasize,k=-0.6):
        dflist = []
        prodlist = []
        To = 10
        df = 0.0
        prod = 1.0
        def lam(s):
            return (1.0 - 1.0/((s-2)*k+To))
        for t in reversed(range(1,datasize+1)):
            prod = 1.0
            if t<datasize: prod = prodlist[-1]*lam(t)
            prodlist.append(prod)
            if t<datasize: df = dflist[-1]+prod
            else: df = prod
            dflist.append(df)
            self.learningrates.append(1.0/df)
        self.set_learning_rates2(datasize,k)

    def perform_updates(self,learningrate,datasize,sentence_count):
        self.sentence_count += 1
        
        # Refreshing catcounts that are not refreshed otherwise
        if self.sentence_count > 1:
            for syn_key in self.catcounts:
                update = -self.catcounts[syn_key]*learningrate
                self.catcounts[syn_key] += update

            self.sem_distribution.refresh(learningrate)
        
        # UPDATING PARAMETERS, without the refreshing substraction part
        for l in self.updates:
            li = self.lex[l]
            self.refresh_params(li.word,li.syn,li.sem_key,sentence_count)
            update = 0.0
            update += self.updates[l]*learningrate*datasize
            self.lex[l].increment_num_seen()
            self.update_params(l[0],l[1],l[2],update,sentence_count)
        
        # PRUNING STAGE
        todel = []
        del_space = 10
        delfrac = 0.1
        minsemalph = 0.001
        
        if sentence_count%del_space==0 and sentence_count > 1:
            for swk in self.sem_to_word:
                sw = self.sem_to_word[swk]
                self.refresh_sem_word_params(sw.sem_key,sw.word)
                semalph = self.sem_distribution.sum_alphas(sw.sem_key)
                swalph = sw.alpha
                if semalph == 0.0 or semalph < minsemalph or swalph < 0.0 or swalph / semalph < delfrac:
                    todel.append(swk)

        print('Deleting '+str(len(todel))+' entries of the total '+str(len(self.sem_to_word)))
                  
        for swk in todel:
            sw = self.sem_to_word[swk]
            for li in sw.lex_items:
                lk = (li.word,li.syn,li.sem_key)
                self.refresh_params(li.word,li.syn,li.sem_key,self.sentence_count)
                i = 0
                for l2 in self.words[li.word]:
                    if l2==lk: del self.words[li.word][i]
                    i += 1
                del self.lex[lk]
                self.sem_distribution.update_alpha(li.syn,self.semtoshell[li.sem_key],li.sem_key,-li.alpha)
            del self.sem_to_word[swk]
          
    @staticmethod
    def set_one_word(oneWord):
        Lexicon.mwe = not oneWord
    
    @staticmethod
    def log_word_prior(word):
        numbound = word.count(" ")
        numchar = len(word) - numbound
        log_prior = numchar*log(0.5) - 4.0*numbound
        return log_prior
    
    def clear_updates(self):
        self.updates = {}
    
    def getMaxWordForSynSem(self,syn_key,sem_key):
        """
        Returns the maximum likelihood word for this (syn,sem) pair.
        """
        relevant_key_vals = [(k,v.alpha) for k,v in self.lex.items() if k[1] == syn_key and k[2] == sem_key]
        if relevant_key_vals == []:
            return None
        else:
            return max(relevant_key_vals,key=lambda x: x[1])[0]

