###########################################
# Classes USED TO CREATE THE GRAMMAR      #
###########################################

import copy
import random
import cat
from math import exp
from math import log
from scipy.special import psi
from tools import inf
from tools import log_sum
from errorFunct import error

hacked_prior = True  # The hack introduced by Omri Abend, 25/5/15 to make all 6 transitive
                     # verb categories possible

###########################################
# One rule for each syntactic category    #
# which defines how it can expand into one#
# of a set of targets.                    #
###########################################
class Rule:
    def __init__(self,rule_head,alpha_top,beta_tot,beta_lex):
        self.Rule_Head = rule_head
        self.Targets = {}
        
        # sort these out #
        self.alpha_top = alpha_top
        self.alpha_tot = 0.0
        self.beta_tot = beta_tot
        self.beta_lex = beta_lex

        # sort these out #
        self.temp_alpha_top = alpha_top
        self.temp_alpha_tot = 0.0
        self.temp_beta_tot = beta_tot
        self.temp_beta_lex = beta_lex

        ##################
        
        # these are for  #
        # the i/o alg     #
        #self.top_term = 0.0
        self.bottom_term = 0.0
        
        self.Targets[self.Rule_Head+'_LEX'] = Target(self.Rule_Head,'LEX')

        correction = 1.0
        if hacked_prior and self.Rule_Head != 'START':
            directions = cat.all_directions(cat.synCat.readCat(self.Rule_Head))
            if len(directions) >= 2:
                if directions[-1] == directions[-2]:
                    correction = 4.0 / 3
                else:
                    correction = 2.0 / 3
        self.Targets[self.Rule_Head+'_LEX'].prior = 0.5 * correction

    def set_temp_params(self):
        self.temp_alpha_top = self.alpha_top
        self.temp_alpha_tot = self.alpha_top
        self.temp_beta_tot = self.beta_tot
        self.temp_beta_lex = self.beta_lex
    @staticmethod
    def rule_prior(left_syn,right_syn,direction,numcomp):
        ruleprior = 0.5
        
        if direction=="fwd": ruleprior=ruleprior*right_syn.prior()
        elif direction=="back": ruleprior=ruleprior*left_syn.prior()
        elif direction is None and right_syn is None: ruleprior= ruleprior * left_syn.prior()
        
        if numcomp>0:
            ruleprior = ruleprior*(0.25/numcomp)
        else:
            ruleprior = ruleprior*0.5

        return ruleprior
    
    def check_target(self,left_syn,right_syn,direction,numcomp):
        left_synstring = left_syn.toString()
        if self.Rule_Head=="START" :
            if not self.Targets.has_key(left_syn):
                self.Targets[left_synstring] = Target(self.Rule_Head,(left_synstring,None))
                self.Targets[left_synstring].prior = \
                    Rule.rule_prior(left_syn,right_syn,direction,numcomp)
            return
        right_synstring = right_syn.toString()
        t = left_synstring+'#####'+right_synstring
        if not self.Targets.has_key(t):
            self.Targets[t] = Target(self.Rule_Head,(left_synstring,right_synstring))
            self.Targets[t].prior = Rule.rule_prior(left_syn,right_syn,direction,numcomp)
            #print "setting prior to ",self.Targets[t].prior," for ",t,direction,numcomp
            # Need to work out actual rule prior
            # self.alpha_tot = self.alpha_tot + 1.0
            # self.beta_tot = self.beta_tot+1.0
            # self.temp_alpha_tot = self.temp_alpha_tot + 1.0
            # self.temp_beta_tot = self.temp_beta_tot+1.0
        
            # self.Targets[t].increment_alpha(1.0)
            # at the moment, just adding 1 to CRP
            # doing nothing for words though

    #######################################
    # kinda like up to here           #
    #######################################
    def update_bottom(self,a_pq,B_pq):
        self.bottom_term += a_pq*B_pq

    def update_temp_params(self,target,prob):
        print "this is ",self
        print "temp beta tot is ",self.temp_beta_tot
        print "temp beta lex is ",self.temp_beta_lex
        print "beta tot is ",self.beta_tot
        print "beta lex is ",self.beta_lex
        print "for ",self.Rule_Head+'_LEX'
        print "prob is ",prob
        error("this is gonna be hard")
        self.temp_beta_tot = prob + self.beta_tot

        if target == self.Rule_Head+'_LEX':
            self.temp_beta_lex = prob + self.beta_lex
        else: 
            self.temp_alpha_tot = prob + self.alpha_top
            self.Targets[target].increment_temp_alpha(prob)
    

    def set_temp_params(self,target):
        self.temp_beta_tot = self.beta_tot 
        if target == self.Rule_Head+'_LEX':
            self.temp_beta_lex = self.beta_lex
        else: 
            self.temp_alpha_tot = self.alpha_tot
            self.Targets[target].set_temp_alpha()
    

    def update_params(self,target,prob,learningrate,datasize,gamma):
        verbose = False
        if verbose:
            print "param update is ",prob," for ",target
            print "learning rate is ",learningrate
            print "gamma is ",gamma
            print "alpha is ",self.Targets[target].alpha
        #self.beta_tot += (prob*learningrate)/gamma # - learningrate*self.
        #if True==False and target == self.Rule_Head+'_LEX':
        #self.beta_lex += (prob*learningrate)/gamma - learningrate*self.beta_lex/gamma
#self.beta_tot -= learningrate*self.beta_lex
#        if True: 
            #self.beta_tot -= learningrate*self.Targets[target].alpha
        self.alpha_tot += ((prob*learningrate)/gamma)*datasize - learningrate*self.Targets[target].alpha
        if verbose: print "updating target alpha by ",((prob*learningrate)/gamma)*datasize - learningrate*self.Targets[target].alpha
        self.Targets[target].increment_alpha(((prob*learningrate)/gamma)*datasize  - learningrate*self.Targets[target].alpha)
    
    def update_log_params(self,target,log_prob):
        prob = exp(log_prob)
        self.beta_tot += prob
        if target == self.Rule_Head+'_LEX':
            self.beta_lex += prob
        else: self.alpha_tot += prob
        self.Targets[target].increment_alpha(prob)
            
    def return_prob(self,target,sentence_count):
        return exp(self.return_log_prob(target,sentence_count))

    def return_log_prob(self,target,sentence_count):
        verbose = False
        if sentence_count != 0:
            scale = Rules.extrascale*sentence_count
        else:
            scale = 1.0
        scale = 1.0            
        log_prior = log(self.Targets[target].prior)
        unseen = log_prior + psi(scale*self.alpha_top)-psi(scale*self.alpha_tot+scale*self.alpha_top)        

        a_t = scale*self.Targets[target].alpha
        p1 = psi(a_t)
        p2 = psi(scale*self.alpha_tot+scale*self.alpha_top)
        log_seen = p1-p2

        if a_t == 0.0: return unseen
        log_prob = log_sum(log_seen,unseen)
        if verbose:
            print "\nfor ",target
            print "alpha is ",self.Targets[target].alpha
            print "alpha tot is ",self.alpha_tot
            print "alpha o is ",self.alpha_top
            print "seen log prob is ",log_seen
            print "unseen log prob is ",unseen
            print "log prob is ",log_prob
            print "scale is ",scale
        return log_prob

    def return_map_log_prob(self,target,sentence_count):
        verbose = False
        if sentence_count != 0:
            scale = Rules.extrascale*sentence_count
        else:
            scale = 1.0
        scale = 1.0            
        log_prior = log(self.Targets[target].prior)
        unseen = log_prior + log(scale*self.alpha_top) - \
        log(scale*self.alpha_tot+scale*self.alpha_top)        

        a_t = scale*self.Targets[target].alpha
        
        #p1 = psi(a_t)
        #p2 = psi(scale*self.alpha_tot+scale*self.alpha_top)
        #log_seen = p1-p2
        if a_t <= 10E-100:
            print "a_t is ",a_t
            return unseen
        log_seen = log(a_t) - log(scale * self.alpha_tot + scale * self.alpha_top)
        log_prob = log_sum(log_seen,unseen)
        if verbose:
            print "\nfor ",target
            print "alpha is ",self.Targets[target].alpha
            print "alpha tot is ",self.alpha_tot
            print "alpha o is ",self.alpha_top
            print "seen log prob is ",log_seen
            print "unseen log prob is ",unseen
            print "log prob is ",log_prob
            print "scale is ",scale
        return log_prob

    def check_alpha_tot(self):  
        pass
        
    def check_alpha_tot2(self):
        at = 0
        for t in self.Targets:
            at+=self.Targets[t].alpha
        if not at+10E-5>self.alpha_tot>at-10E-5:
            print "at = ",at
            print "alpha tot = ",self.alpha_tot 
            error("alpha_tot2 error")

    def return_temp_prob(self,target):
        # keep in logspace
        # really need to work out how this actually deals
        # with new rules
        # where is the prior????
        #print "temp beta lex is ",self.temp_beta_lex
        #print "beta lex is ",self.beta_lex
        #print "temp beta tot is ",self.temp_beta_tot
        #print "beta tot is ",self.beta_tot
        #print "this is ",self
        prior = 0.01
        if not self.Targets.has_key(target): return prior
        if target == self.Rule_Head+'_LEX':
            prob = exp(psi(self.temp_beta_lex))/exp(psi(self.temp_beta_tot))
        else:
            a_t = prior
            if self.Targets.has_key(target): a_t = max(self.Targets[target].temp_alpha,prior)
#            print self.beta_tot,  self.beta_lex
            prob = exp(psi(self.temp_beta_tot - self.beta_lex))/exp(psi(self.temp_beta_tot))
            p1 = psi(a_t)
            p2 = psi(self.temp_alpha_tot)
            #print "p1 = ",p1," p2 = ",p2
            prob = prob*exp(p1-p2)#psi(a_t) - psi(self.temp_alpha_tot))            
        if prob > 1.0:
            print 'rule prob over 1 for ',target, self.Targets[target].temp_alpha, self.temp_alpha_tot, prob
            print self.temp_beta_lex,self.temp_beta_tot
        return prob
        
    def clear_probs(self):
        for t in self.Targets:
            self.Targets[t].clear_probs()
        self.bottom_term = 0

###########################################

###########################################
# Targets are pairs of syntactic categories
# that come from a rule_head.             #
###########################################
class Target:
    def __init__(self, Rule_Head, t):
        self.Rule_Head = Rule_Head
        self.prior = None
        if t == 'LEX':
            self.key = self.Rule_Head+'_LEX'
            self.Left_Rep = None
            self.Right_Rep = None
        else:
            self.Left_Rep = t[0]
            self.Right_Rep = t[1]
            if t[1]: self.key = self.Left_Rep+'#####'+self.Right_Rep
            else: self.key = self.Left_Rep
        self.alpha = 0.0
        self.temp_alpha = 0.0
        self.top_term = 0.0
        self.old_prob = 1.0
    def return_key(self):
        return self.key

    def increment_temp_alpha(self,a):
        self.temp_alpha = self.alpha+a
    def set_temp_alpha(self):
        self.temp_alpha = self.alpha

    def increment_alpha(self,a):
        self.alpha = self.alpha+a

    def clear_probs(self):
        self.top_term = 0

###########################################





###########################################
# Holds all of the Rules.                 #
###########################################
class Rules:
    extrascale = 1.0
    def __init__(self,alpha_top,beta_tot,beta_lex):
        self.usegamma = True
        self.updateweight = 0.1
        self.alpha_top = alpha_top
        self.orig_alpha_top = alpha_top
        self.beta_tot = beta_tot
        self.beta_lex = beta_lex
        self.Rules = {}
        self.Syntactic_Categories = {}
        self.Start_Targets = Start_Rule()
        self.sentence_count = 0
        # Targets point to the syntactic head #
        self.Targets = {}
        self.updates = {}
        self.Rules["START"] = Rule("START",self.alpha_top,self.beta_tot,self.beta_lex)
    def check_start_rule(self,rule_head):
        c = rule_head.toString()
        if self.Targets.has_key(c): return
        self.Rules["START"].check_target(rule_head,None,None,0)
        self.Targets[c] = "START"
        print "adding ",c," to START"
    def check_rule(self,rule_head,left_syn,right_syn,direction,numcomp):
        if not self.Rules.has_key(rule_head):
            self.Rules[rule_head] = Rule(rule_head,self.alpha_top,self.beta_tot,self.beta_lex)
            self.Targets[rule_head+'_LEX'] = rule_head
        if left_syn is not None and right_syn is not None:
            self.Rules[rule_head].check_target(left_syn,right_syn,direction,numcomp)
            if not self.Targets.has_key(left_syn.toString()+'#####'+right_syn.toString()):
                self.Targets[left_syn.toString()+'#####'+right_syn.toString()] = rule_head

    def check_target(self,t):
        if self.Targets.has_key(t):
            return True
        else:
            return None

    def update_bottom(self,rule_head,a_pq,B_pq):
        self.Rules[rule_head].update_bottom(a_pq,B_pq)

    def set_temp_params(self,target):
        r = self.Targets[target]
        self.Rules[r].set_temp_params(target)

    def update_log_params(self,target,log_prob):
        r = self.Targets[target]
        self.Rules[r].update_log_params(target,log_prob)

    def store_log_update(self,target,log_prob):
        if log_prob > 10E-4:
            print "logprob is ",log_prob," for ",target
            error("logprob is too large")
        if log_prob > 0.0: log_prob = 0.0
        prob = exp(log_prob)
        if not self.updates.has_key(target): self.updates[target]=prob
        else: self.updates[target] += prob
            
    def perform_updates(self,learningrate,datasize,sentence_count):
        gamma = self.orig_alpha_top + datasize
        if not self.usegamma: gamma = 1.0
        if sentence_count > 0:
            alpha_top_update = learningrate*self.alpha_top*(1.0/gamma - 1.0) 
            self.alpha_top = self.alpha_top + alpha_top_update       
        
        for target in self.Targets: #updates:
            r = self.Targets[target]
            prob = 0.0
            if self.updates.has_key(target): prob = self.updates[target]
            self.Rules[r].update_params(target,prob,learningrate,datasize,gamma)
        alphamin = -inf #10E3
        todel = []
        for r in self.Rules:
            self.Rules[r].alpha_top = self.alpha_top
            self.Rules[r].check_alpha_tot()
            if self.Rules[r].alpha_tot < alphamin:
                for t in self.Rules[r].Targets:
                    del self.Targets[t]
                todel.append(r)
        for r in todel:
            print "deleting rule ",r
            del self.Rules[r]
        self.sentence_count = sentence_count
        
    def set_temp_params(self):
        for target in self.updates:
            r = self.Targets[target]
            self.Rules[r].set_temp_params(target)
            
    def perform_temp_updates(self):
        for target in self.updates:
            r = self.Targets[target]
            prob = self.updates[target]
            self.Rules[r].update_temp_params(target,prob)

    def clear_updates(self):
        self.updates = {}

    def return_prob(self,head,target):
        prob = self.Rules[head].return_prob(target,self.sentence_count)
        return prob

    def return_map_log_prob(self,head,target):
        if head == "START" and not self.Targets.has_key(target):
            print "in get log prob and not got ",target
            self.check_start_rule(target)
        logprior = log(self.Rules[head].Targets[target].prior)
        if self.Rules.has_key(head):
            log_prob = self.Rules[head].return_map_log_prob(target,self.sentence_count)
        else:
            return logprior
        return log_prob

    def return_map_prob_distribution(self,head):
        if not self.Rules.has_key(head):
            return []
        else:
            D = []
            tot = 0.0
            for target in self.Rules[head].Targets.keys():
                cats = tuple(target.split('#####'))
                val = exp(self.return_map_log_prob(head,target))
                D.append((cats,val))
                tot += val
            D = sorted([(x[0],x[1]/tot) for x in D],key=lambda x: -x[1])
            return D
            
    def return_leaf_map_log_prob(self,head):
        """
        Written by Omri Abend 28/7
        Returns the log MAP probability of generating
        a leaf from the category head. 

        Input:
        head - a string with the syntactic category

        Output:
        a double which is the log MAP probability. None if head is not a category
        in the grammar.
        """
        if not self.Rules.has_key(head):
            return None
        else:
            return self.return_map_log_prob(head,head+'_LEX')

    def return_log_prob(self,head,target):
        if head == "START" and not self.Targets.has_key(target):
            print "in get log prob and not got ",target
            self.check_start_rule(target)

        logprior = log(self.Rules[head].Targets[target].prior)

        if self.Rules.has_key(head):
            log_prob = self.Rules[head].return_log_prob(target,self.sentence_count)
        else:
            return logprior
        return log_prob    

    def update_alphas(self):
        for r in self.Rules:
            if self.Rules[r].bottom_term > 0:
                self.Rules[r].update_alphas(self.updateweight)
    def compare_probs(self):
        converged = True
        for r in self.Rules:
            rc = self.Rules[r].compare_probs()
            if not rc:
                converged = False
        return converged
        
    def clear_probs(self):
        for r in self.Rules:
            self.Rules[r].clear_probs()
###########################################


###########################################
# Special Rule for going from the START   #
# symbol to the top category.             #
###########################################
class Start_Rule:
    def __init__(self):
        self.Targets = {}
        self.total_count = 0
        self.alpha_top = 10 ## this could be changed for a diff learning rate
        #self.atomic_types = [ 'PP' , 'VP_[to]' , 'NP_[SUBJ]' , 'S_[y/n]' , 'NP_[OBJ]' , 'NP_[POBJ]' , 'NP_[PRED]' , 'N' , 'NP' , 'NP_[OBJ2]' , 'VP_[perf]' , 'S_[dcl]' , 'VP_[ing]' , 'S_[emb]' , 'VP_[b]' , 'S_[wh]' ]


    def increment_target(self, Target):
        ##want to add one for each Target seen
        if self.Targets.has_key(Target):
            self.Targets[Target].increment()
            self.total_count += 1
            #return True
        else:
            self.Targets[Target] = Start_Target(Target)
            self.Targets[Target].increment()
            self.total_count += 1
            
    def return_prob(self,Target):
        print "getting start prob for ",Target
        if self.Targets.has_key(Target):
            #print 'has key, count is ',self.Targets[Target].count
            prob = float(self.Targets[Target].count)/(self.total_count+1)
            return prob
        else:
            prob = float(1)/(self.total_count+1)
            return prob
        
    def return_unk_prob(self):
        prob = self.alpha_top/(self.total_count+self.alpha_top)
        return prob
###########################################


###########################################
# Start rule has to point to something.   #
###########################################
class Start_Target:
    def __init__(self,Target):
        self.Head = Target
        self.count = 0
    def increment(self):
        self.count += 1
    
#########################################
#########################################
    


###########################################
