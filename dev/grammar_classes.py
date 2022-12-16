###########################################
# Classes USED TO CREATE THE GRAMMAR      #
###########################################
import cat
from math import exp
from math import log
from scipy.special import psi
from tools import inf
from tools import log_sum

hacked_prior = True  # The hack introduced by Omri Abend, 25/5/15 to make all 6 transitive
                     # verb categories possible

# One rule for each syntactic category    #
# which defines how it can expand into one#
# of a set of targets.                    #
class Rule:
    def __init__(self, rule_head, alpha_top, beta_tot, beta_lex):
        self.rule_head = rule_head
        self.targets = {}

        self.alpha_top = alpha_top
        self.alpha_tot = 0.0
        self.beta_tot = beta_tot
        self.beta_lex = beta_lex

        # for the i/o alg
        self.bottom_term = 0.0

        self.targets[self.rule_head+'_LEX'] = Target(self.rule_head, 'LEX')

        correction = 1.0
        if hacked_prior and self.rule_head != 'START':
            directions = cat.all_directions(cat.SynCat.read_cat(self.rule_head))
            if len(directions) >= 2:
                if directions[-1] == directions[-2]:
                    correction = 4.0 / 3
                else:
                    correction = 2.0 / 3
        self.targets[self.rule_head+'_LEX'].prior = 0.5 * correction

    @staticmethod
    def rule_prior(left_syn, right_syn, direction, numcomp):
        ruleprior = 0.5

        if direction=="fwd": ruleprior=ruleprior*right_syn.prior()
        elif direction=="back": ruleprior=ruleprior*left_syn.prior()
        elif direction is None and right_syn is None: ruleprior= ruleprior * left_syn.prior()

        if numcomp>0:
            ruleprior = ruleprior*(0.25/numcomp)
        else:
            ruleprior = ruleprior*0.5

        return ruleprior

    def check_target(self, left_syn, right_syn, direction, numcomp):
        left_synstring = left_syn.to_string()
        if self.rule_head=="START":
            if left_syn not in self.targets:
                self.targets[left_synstring] = Target(self.rule_head, (left_synstring, None))
                self.targets[left_synstring].prior = \
                    Rule.rule_prior(left_syn, right_syn, direction, numcomp)
            return
        right_synstring = right_syn.to_string()
        t = left_synstring+'#####'+right_synstring
        if t not in self.targets:
            self.targets[t] = Target(self.rule_head, (left_synstring, right_synstring))
            self.targets[t].prior = Rule.rule_prior(left_syn, right_syn, direction, numcomp)

    def update_bottom(self, a_pq, B_pq):
        self.bottom_term += a_pq*B_pq

    def update_params(self, target, prob, learningrate, datasize, gamma):
        verbose = False
        if verbose:
            print("param update is ", prob, " for ", target)
            print("learning rate is ", learningrate)
            print("gamma is ", gamma)
            print("alpha is ", self.targets[target].alpha)
        #self.beta_tot += (prob*learningrate)/gamma # - learningrate*self.
        #if True==False and target == self.rule_head+'_LEX':
        #self.beta_lex += (prob*learningrate)/gamma - learningrate*self.beta_lex/gamma
#self.beta_tot -= learningrate*self.beta_lex
#        if True:
            #self.beta_tot -= learningrate*self.targets[target].alpha
        self.alpha_tot += ((prob*learningrate)/gamma)*datasize - learningrate*self.targets[target].alpha
        if verbose: print("updating target alpha by ", ((prob*learningrate)/gamma)*datasize - learningrate*self.targets[target].alpha)
        self.targets[target].increment_alpha(((prob*learningrate)/gamma)*datasize - learningrate*self.targets[target].alpha)

    def update_log_params(self, target, log_prob):
        prob = exp(log_prob)
        self.beta_tot += prob
        if target == self.rule_head+'_LEX':
            self.beta_lex += prob
        else: self.alpha_tot += prob
        self.targets[target].increment_alpha(prob)

    def return_prob(self, target, sentence_count):
        return exp(self.return_log_prob(target, sentence_count))

    def return_log_prob(self, target, sentence_count):
        verbose = False
        scale = 1.0
        log_prior = log(self.targets[target].prior)
        unseen = log_prior + psi(scale*self.alpha_top)-psi(scale*self.alpha_tot+scale*self.alpha_top)

        a_t = scale*self.targets[target].alpha
        p1 = psi(a_t)
        p2 = psi(scale*self.alpha_tot+scale*self.alpha_top)
        log_seen = p1-p2

        if a_t == 0.0: return unseen
        log_prob = log_sum(log_seen, unseen)
        if verbose:
            print("\nfor ", target)
            print("alpha is ", self.targets[target].alpha)
            print("alpha tot is ", self.alpha_tot)
            print("alpha o is ", self.alpha_top)
            print("seen log prob is ", log_seen)
            print("unseen log prob is ", unseen)
            print("log prob is ", log_prob)
            print("scale is ", scale)
        return log_prob

    def return_map_log_prob(self, target, sentence_count):
        verbose = False
        scale = 1.0
        log_prior = log(self.targets[target].prior)
        unseen = log_prior + log(scale*self.alpha_top) - \
        log(scale*self.alpha_tot+scale*self.alpha_top)

        a_t = scale*self.targets[target].alpha

        if a_t <= 10E-100:
            return unseen
        log_seen = log(a_t) - log(scale * self.alpha_tot + scale * self.alpha_top)
        log_prob = log_sum(log_seen, unseen)
        if verbose:
            print("\nfor ", target)
            print("alpha is ", self.targets[target].alpha)
            print("alpha tot is ", self.alpha_tot)
            print("alpha o is ", self.alpha_top)
            print("seen log prob is ", log_seen)
            print("unseen log prob is ", unseen)
            print("log prob is ", log_prob)
            print("scale is ", scale)
        return log_prob

    def check_alpha_tot2(self):
        at = 0
        for t in self.targets:
            at+=self.targets[t].alpha
        if not at+10E-5>self.alpha_tot>at-10E-5:
            print("at = ", at)
            print("alpha tot = ", self.alpha_tot)

    def return_temp_prob(self, target):
        # really need to work out how this actually deals
        # with new rules, where is the prior????
        prior = 0.01
        if target not in self.targets: return prior
        if target == self.rule_head+'_LEX':
            prob = exp(psi(self.temp_beta_lex))/exp(psi(self.temp_beta_tot))
        else:
            a_t = prior
            if target in self.targets: a_t = max(self.targets[target].temp_alpha, prior)
            prob = exp(psi(self.temp_beta_tot - self.beta_lex))/exp(psi(self.temp_beta_tot))
            p1 = psi(a_t)
            p2 = psi(self.temp_alpha_tot)
            prob = prob*exp(p1-p2)
        if prob > 1.0:
            print('rule prob over 1 for ', target, self.targets[target].temp_alpha, self.temp_alpha_tot, prob)
            print(self.temp_beta_lex, self.temp_beta_tot)
        return prob

    def clear_probs(self):
        for t in self.targets:
            self.targets[t].clear_probs()
        self.bottom_term = 0

# targets are pairs of syntactic categories
# that come from a rule_head.             #
class Target:
    def __init__(self, rule_head, t):
        self.rule_head = rule_head
        self.prior = None
        if t == 'LEX':
            self.key = self.rule_head+'_LEX'
            self.left_rep = None
            self.right_rep = None
        else:
            self.left_rep = t[0]
            self.right_rep = t[1]
            if t[1]: self.key = self.left_rep+'#####'+self.right_rep
            else: self.key = self.left_rep
        self.alpha = 0.0
        self.temp_alpha = 0.0
        self.top_term = 0.0
        self.old_prob = 1.0

    def return_key(self):
        return self.key

    def increment_temp_alpha(self, a):
        self.temp_alpha = self.alpha+a

    def set_temp_alpha(self):
        self.temp_alpha = self.alpha

    def increment_alpha(self, a):
        self.alpha = self.alpha+a

    def clear_probs(self):
        self.top_term = 0

class Rules:
    def __init__(self, alpha_top, beta_tot, beta_lex, usegamma=False):
        self.usegamma = usegamma
        self.updateweight = 0.1
        self.alpha_top = alpha_top
        self.orig_alpha_top = alpha_top
        self.beta_tot = beta_tot
        self.beta_lex = beta_lex
        self.rules = {}
        self.sentence_count = 0
        # targets point to the syntactic head #
        self.targets = {}
        self.updates = {}
        self.rules["START"] = Rule("START", self.alpha_top, self.beta_tot, self.beta_lex)

    def check_start_rule(self, rule_head):
        c = rule_head.to_string()
        if c in self.targets: return
        self.rules["START"].check_target(rule_head, None, None, 0)
        self.targets[c] = "START"

    def check_rule(self, rule_head, left_syn, right_syn, direction, numcomp):
        if rule_head not in self.rules:
            self.rules[rule_head] = Rule(rule_head, self.alpha_top, self.beta_tot, self.beta_lex)
            self.targets[rule_head+'_LEX'] = rule_head
        if left_syn is not None and right_syn is not None:
            self.rules[rule_head].check_target(left_syn, right_syn, direction, numcomp)
            if left_syn.to_string()+'#####'+right_syn.to_string() not in self.targets:
                self.targets[left_syn.to_string()+'#####'+right_syn.to_string()] = rule_head

    def check_target(self, t):
        if t in self.targets:
            return True
        else:
            return None

    def update_bottom(self, rule_head, a_pq, B_pq):
        self.rules[rule_head].update_bottom(a_pq, B_pq)

    def update_log_params(self, target, log_prob):
        r = self.targets[target]
        self.rules[r].update_log_params(target, log_prob)

    def store_log_update(self, target, log_prob):
        if log_prob > 10E-4:
            print("logprob is too large")
        if log_prob > 0.0: log_prob = 0.0
        prob = exp(log_prob)
        if target not in self.updates: self.updates[target]=prob
        else: self.updates[target] += prob

    def perform_updates(self, learningrate, datasize, sentence_count):
        gamma = self.orig_alpha_top + datasize
        if not self.usegamma: gamma = 1.0
        if sentence_count > 0:
            alpha_top_update = learningrate*self.alpha_top*(1.0/self.gamma - 1.0)
            self.alpha_top = self.alpha_top + alpha_top_update

        for target in self.targets: #updates:
            r = self.targets[target]
            prob = 0.0
            if target in self.updates: prob = self.updates[target]
            self.rules[r].update_params(target, prob, learningrate, datasize, gamma)
        alphamin = -inf #10E3
        todel = []
        for r in self.rules:
            self.rules[r].alpha_top = self.alpha_top
            if self.rules[r].alpha_tot < alphamin:
                for t in self.rules[r].targets:
                    del self.targets[t]
                todel.append(r)
        for r in todel:
            print("deleting rule ", r)
            del self.rules[r]
        self.sentence_count = sentence_count

    def perform_temp_updates(self):
        for target in self.updates:
            r = self.targets[target]
            prob = self.updates[target]
            self.rules[r].update_temp_params(target, prob)

    def clear_updates(self):
        self.updates = {}

    def return_prob(self, head, target):
        prob = self.rules[head].return_prob(target, self.sentence_count)
        return prob

    def return_map_log_prob(self, head, target):
        if head == "START" and target not in self.targets:
            print("in get log prob and not got ", target)
            self.check_start_rule(target)
        logprior = log(self.rules[head].targets[target].prior)
        if head in self.rules:
            log_prob = self.rules[head].return_map_log_prob(target, self.sentence_count)
        else:
            return logprior
        return log_prob

    def return_map_prob_distribution(self, head):
        if head not in self.rules:
            return []
        else:
            D = []
            tot = 0.0
            for target in list(self.rules[head].targets.keys()):
                cats = tuple(target.split('#####'))
                val = exp(self.return_map_log_prob(head, target))
                D.append((cats, val))
                tot += val
            D = sorted([(x[0], x[1]/tot) for x in D], key=lambda x: -x[1])
            return D

    def return_leaf_map_log_prob(self, head):
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
        if head not in self.rules:
            return None
        else:
            return self.return_map_log_prob(head, head+'_LEX')

    def return_log_prob(self, head, target):
        if head == "START" and target not in self.targets:
            print("in get log prob and not got ", target)
            self.check_start_rule(target)

        logprior = log(self.rules[head].targets[target].prior)

        if head in self.rules:
            log_prob = self.rules[head].return_log_prob(target, self.sentence_count)
        else:
            return logprior
        return log_prob

    def update_alphas(self):
        for r in self.rules:
            if self.rules[r].bottom_term > 0:
                self.rules[r].update_alphas(self.updateweight)

    def compare_probs(self):
        converged = True
        for r in self.rules:
            rc = self.rules[r].compare_probs()
            if not rc:
                converged = False
        return converged

    def clear_probs(self):
        for r in self.rules:
            self.rules[r].clear_probs()

# Special rule for going from the START   #
# symbol to the top category.             #
class Start_Rule:
    def __init__(self):
        self.targets = {}
        self.total_count = 0
        self.alpha_top = 10 ## this could be changed for a diff learning rate
        #self.atomic_types = [ 'PP' , 'VP_[to]' , 'NP_[SUBJ]' , 'S_[y/n]' , 'NP_[OBJ]' , 'NP_[POBJ]' , 'NP_[PRED]' , 'N' , 'NP' , 'NP_[OBJ2]' , 'VP_[perf]' , 'S_[dcl]' , 'VP_[ing]' , 'S_[emb]' , 'VP_[b]' , 'S_[wh]' ]

    def increment_target(self, target):
        ##want to add one for each target seen
        if target in self.targets:
            self.targets[target].increment()
            self.total_count += 1
        else:
            self.targets[target] = Start_Target(target)
            self.targets[target].increment()
            self.total_count += 1

    def return_prob(self, target):
        if target in self.targets:
            prob = float(self.targets[target].count)/(self.total_count+1)
            return prob
        else:
            prob = float(1)/(self.total_count+1)
            return prob

    def return_unk_prob(self):
        prob = self.alpha_top/(self.total_count+self.alpha_top)
        return prob

# Start rule has to point to something.   #
class Start_Target:
    def __init__(self, target):
        self.Head = target
        self.count = 0
    def increment(self):
        self.count += 1
