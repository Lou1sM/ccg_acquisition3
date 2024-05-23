import numpy as np
from dl_utils.misc import check_dir
from gt_parse_graphs import gts
import os
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint
from copy import copy
from dl_utils.misc import set_experiment_dir
from os.path import join
import pandas as pd
from utils import file_print, get_combination, is_direct_congruent, combine_lfs, logical_type_raise, maybe_de_type_raise, possible_syn_cats, infer_slash, lf_cat_congruent, lf_acc, split_respecting_brackets
from errors import CCGLearnerError
from time import time
import argparse
from abc import ABC
import re
from parser import LogicalForm, ParseNode
import json
from learner_config import all_gt_lexicons


def type_raise(cat,direction,out_cat='S'):
    if direction == 'fwd':
        return f'{out_cat}/({out_cat}\\{cat})'
    elif direction == 'bck':
        return f'{out_cat}\\({out_cat}/{cat})'
    elif direction == 'sem':
        return f'{out_cat}|({out_cat}|{cat})'

def get_root(n):
    root = n
    while True:
        if root.parent is None:
            break
        else:
            root = root.parent
    return root

class DirichletProcess():
    def __init__(self,alpha):
        self.memory = {'COUNT':0}
        self.alpha = alpha

    def observe(self,obs,weight):
        if obs in self.memory:
            self.memory[obs] += weight
        else:
            self.memory = self.memory | {obs:weight}
        self.memory['COUNT'] += weight
        assert np.allclose(2*self.memory['COUNT'],sum(self.memory.values()),rtol=1e-6)

    def prob(self,obs, ignore_prior=False):
        base_prob = 0.25 * (1/12)**(obs.count('/') + obs.count('\\') + obs.count('|'))
        mx = self.memory.get(obs,0)
        count = self.memory['COUNT']
        return mx/count if ignore_prior else (mx+self.alpha*base_prob)/(count+self.alpha)

    def top_k(self, k=10):
        return sorted([(k,self.prob(k)) for k in self.memory.keys() if k not in ('X','COUNT')],key=lambda x:x[1],reverse=True)[:k]

class BaseDirichletProcessLearner(ABC):
    def __init__(self,alpha):
        self.alpha = alpha
        self.base_alpha = alpha
        self.eval_alpha = ARGS.eval_alpha
        self.memory = {}
        self.base_distribution_cache = {}
        self.marg_prob_cache = {}
        self.buffers = []
        self.is_training = True

    def train(self):
        self.alpha = self.base_alpha
        self.is_training = True

    def eval(self):
        self.alpha = self.eval_alpha
        self.is_training = False

    def base_distribution_(self,x):
        raise NotImplementedError

    def set_from_dict(self,dict_to_set_from):
        self.memory = dict_to_set_from['memory']
        self.base_distribution_cache = dict_to_set_from['base_distribution_cache']

    def _prob(self,y,x):
        base_prob = self.base_distribution(y)
        if x not in self.memory:
            return base_prob
        mx = self.memory[x].get(y,0)
        return (mx+self.alpha*base_prob)/(self.memory[x]['COUNT']+self.alpha)

    def marg_prob(self, x, ignore_cache=False):
        if x in self.marg_prob_cache and not ignore_cache:
            return self.marg_prob_cache[x]
        prob = self._marg_prob(x)
        self.marg_prob_cache[x] = prob
        return prob

    def _marg_prob(self, x):
        numerator = sum(self.prob(x, k)*v['COUNT'] for k,v in self.memory.items())
        denominator = sum(v['COUNT'] for k,v in self.memory.items())
        prob = numerator / denominator
        return prob

    def prob(self,*args,**kwargs):
        return self._prob(*args,**kwargs)

    def _observe(self,y,x,weight):
        if x not in self.memory:
            self.memory[x] = {y:weight,'COUNT':weight}
        elif y not in self.memory[x]:
            self.memory[x] = self.memory[x] | {y:weight}
            self.memory[x]['COUNT'] += weight
        else:
            self.memory[x][y] += weight
            self.memory[x]['COUNT'] += weight
        assert np.allclose(2*self.memory[x]['COUNT'],sum(self.memory[x].values()),rtol=1e-6)

    def observe(self,*args,**kwargs):
        """Can be overwritten if necessary."""
        return self._observe(*args,**kwargs)

    def flush_top_buffer(self, lr=1.0):
        top_buffer = self.buffers.pop(0)
        for _ in range(len(top_buffer)):
            y, x, weight = top_buffer.pop()
            self.observe(y, x, weight*lr)
        assert len(top_buffer) == 0

    def inverse_distribution(self,y): # conditional on y
        seen_before = any([y in d for d in self.memory.values()])
        assert seen_before, f'learner hasn\'t seen word \'{y}\' before'
        inverse_distribution = {x:m.get(y,0) for x,m in self.memory.items()}
        return inverse_distribution

    def base_distribution(self,y):
        if y in self.base_distribution_cache:
            return self.base_distribution_cache[y]
        prob = self.base_distribution_(y)
        self.base_distribution_cache[y] = prob
        return prob

    def conditional_sample(self,x):
        if x not in self.memory:
            breakpoint()
        options,unnormed_probs = zip(*[z for z in self.memory[x].items() if z[0]!='COUNT'])
        probs = np.array(unnormed_probs)/sum(unnormed_probs)
        return np.random.choice(options,p=probs)

    def all_inverse_distributions(self,vocab,base):
        seen_probs = pd.DataFrame([self.inverse_distribution(w) for w in vocab],index=vocab)
        unseen = [x for x in base.index if x not in seen_probs.columns]
        not_in_base = [x for x in seen_probs.columns if x not in base.index]
        seen_probs[unseen] = 0 # some items in base but never seen, set their seen prob to 0
        for b in not_in_base: base[b]=0 # should really recompute, just doing this for now
        base /= base.sum()
        a = self.alpha
        df =(seen_probs+a*base)/(seen_probs.to_numpy().sum(axis=1,keepdims=True)+a)
        assert (df.sum(axis=1)<=1+1e-7).all()
        return df

    def show_commons(self, x):
        print(sorted([(k,v) for k,v in self.memory[x].items()], key=lambda x:x[1],reverse=True)[:10])

    def top_k(self,x,k=10):
        return sorted(self.memory[x].items(), key=lambda x:x[1], reverse=True)[:k]

class CCGDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        #if self.is_training:
            #return 1
        #else:
        n_slashes = len(re.findall(r'[\\/\|]',x))
        return (0.5555)*0.9**(n_slashes+1) # Omri 2017 had 0.2

    def observe(self,y,x,weight):
        if y == 'leaf':
            self._observe(y,maybe_de_type_raise(x),weight)
        else:
            self._observe(y,x,weight)

    #@override
    def _marg_prob(self, x):
        denominator = sum(v['COUNT'] for v in self.memory.values())
        base_prob = self.base_distribution(x)
        numerator = sum(v['COUNT'] for k,v in self.memory.items() if k in possible_syn_cats(x))
        return (numerator + self.alpha*base_prob) / (denominator + self.alpha)

    def prob(self,y,x, can_be_weird_svo=False):
        if y != 'leaf':
            outcat, _ = get_combination(*y.split(' + '))
            if outcat is None: #only time this should happen is during inference with crossed cmp
                return 0
            if not is_direct_congruent(outcat,x):
                return 0
        assert x != 'NP + S/NP\\NP' or not self.is_training
        base_prob = self.base_distribution(y)
        if x not in self.memory:
            if y == 'leaf':
                #return base_prob/(self.memory.get('COUNT',0) + 1) #could be +base_prob instead of 1
                return base_prob
            return base_prob
        mx = self.memory[x].get(y,0)
        return (mx+self.alpha*base_prob)/(self.memory[x]['COUNT']+self.alpha)

class ShellMeaningDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        n_vars = len(set(re.findall(r'(?<!lambda )\$\d',x)))
        n_constants = float('const' in x)+float('quant' in x)
        norm_factor = (np.e+1)/(np.e**2 - np.e - 1)
        return norm_factor * np.e**(-2*n_vars - n_constants)

    def prob(self,y,x):
        #if x not in ['N','Swhq'] and len(split_respecting_brackets(x,sep=['/','\\','|'])) != n_lambda_binders(y)+1:
        if not lf_cat_congruent(y,x):
            #print(y,x)
            return 0
        #y = maybe_de_type_raise(y_)
        base_prob = self.base_distribution(y)
        if x not in self.memory:
            return base_prob
        mx = self.memory[x].get(y,0)
        return (mx+self.alpha*base_prob)/(self.memory[x]['COUNT']+self.alpha)

class MeaningDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        n_vars = len(set(re.findall(r'(?<!lambda )\$\d',x)))
        n_constants = len(set([z for z in re.findall(r'[a-z]*',x) if 'lambda' not in z and len(z)>0]))
        norm_factor = (np.e+1)/(np.e**2 - np.e - 1)
        return norm_factor * np.e**(-2*n_vars - n_constants)

class WordSpanDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution(self,x):
        assert len(x) > 0
        if ARGS.remove_vowels:
            return 14**-len(x)
        else:
            return 28**-(len(x)/2) # kinda approximate num phonetic chunks

class LanguageAcquirer():
    def __init__(self, lr, vocab_thresh):
        self.lr = lr
        self.syntaxl = CCGDirichletProcessLearner(10)
        self.shmeaningl = ShellMeaningDirichletProcessLearner(1)
        self.meaningl = MeaningDirichletProcessLearner(1)
        self.wordl = WordSpanDirichletProcessLearner(0.25)
        self.full_lfs_cache = {} # maps lf_strings to LogicalForm objects
        #self.lf_parts_cache = {'splits':{},'cats':{}} # maps LogicalForm objects to lists of (left-child,right-child)
        self.caches = {'splits':{}, 'cats':{}}
        self.parse_node_cache = {} # maps utterances (str) to ParseNode objects, including splits
        self.root_sem_cat_memory = DirichletProcess(1) # counts of the sem_cats it's seen as roots
        self.sem_cat_memory = DirichletProcess(1) # counts of the sem_cats it's seen anywhere
        self.syn_cat_memory = DirichletProcess(1)
        self.lf_memory = DirichletProcess(1)
        self.shell_lf_memory = DirichletProcess(1)
        self.leaf_syncat_memory = DirichletProcess(1)
        self.vocab_thresh = vocab_thresh
        self.beam_size = 50

    def train(self):
        self.syntaxl.train()
        self.shmeaningl.train()
        self.meaningl.train()
        self.wordl.train()

    def eval(self):
        self.syntaxl.eval()
        self.shmeaningl.eval()
        self.meaningl.eval()
        self.wordl.eval()

    @property
    def lf_vocab(self):
        return [k for k,v in self.wordl.memory.items() if v['COUNT']>self.vocab_thresh]

    @property
    def shell_lf_vocab(self):
        return [k for k,v in self.meaningl.memory.items() if v['COUNT']>self.vocab_thresh]

    @property
    def sem_cat_vocab(self):
        return [k for k,v in self.shmeaningl.memory.items() if v['COUNT']>self.vocab_thresh]

    @property
    def syn_cat_vocab(self):
        return [k for k,v in self.syntaxl.memory.items() if v['COUNT']>self.vocab_thresh]

    @property
    def splits_vocab(self):
        return list(set([k for x in self.syntaxl.memory.values() for k,v in x.items() if k!='COUNT' if v>self.vocab_thresh]))

    @property
    def mwe_vocab(self):
        return list(set([k for x in self.wordl.memory.values() for k,v in x.items() if k!='COUNT' if v>self.vocab_thresh]))

    @property
    def vocab(self):
        return list(set([w for x in self.wordl.memory.values()
                        for w in x.keys() if ' ' not in w]))

    def load_from(self,fpath):
        distributions_fpath = join(fpath,'distributions.json')
        with open(distributions_fpath) as f:
            to_load = json.load(f)
        self.syntaxl.set_from_dict(to_load['syntax'])
        self.shmeaningl.set_from_dict(to_load['shmeaningl'])
        self.meaningl.set_from_dict(to_load['meaning'])
        self.wordl.set_from_dict(to_load['word'])
        self.root_sem_cat_memory.memory = to_load['root_sem_cat']
        self.leaf_syncat_memory.memory = to_load['leaf_syn_cat']
        self.sem_cat_memory.memory = to_load['sem_cat']
        self.syn_cat_memory.memory = to_load['syn_cat']
        self.shell_lf_memory.memory = to_load['shell_lf']
        self.lf_memory.memory = to_load['lf']

        self.lf_word_counts = pd.read_pickle(join(fpath,'lf_word_counts.pkl'))
        self.shell_lf_lf_counts = pd.read_pickle(join(fpath,'shell_lf_lf_counts.pkl'))
        self.sem_shell_lf_counts = pd.read_pickle(join(fpath,'sem_shell_lf_counts.pkl'))
        self.sem_word_counts = pd.read_pickle(join(fpath,'sem_word_counts.pkl'))
        self.marginal_syn_counts = pd.read_pickle(join(fpath,'marginal_syn_counts.pkl'))
        self.syn_word_probs = pd.read_pickle(join(fpath,'syn_word_probs.pkl'))

    def save_to(self,fpath):
        to_dump = {'syntax': {'memory':self.syntaxl.memory,
                  'base_distribution_cache':self.syntaxl.base_distribution_cache},
                  'shmeaningl': {'memory':self.shmeaningl.memory,
                  'base_distribution_cache':self.shmeaningl.base_distribution_cache},
                  'meaning': {'memory':self.meaningl.memory,
                  'base_distribution_cache':self.meaningl.base_distribution_cache},
                  'word': {'memory':self.wordl.memory,
                  'base_distribution_cache':self.wordl.base_distribution_cache},
                  'root_sem_cat': self.root_sem_cat_memory.memory,
                  'sem_cat': self.sem_cat_memory.memory,
                  'syn_cat': self.syn_cat_memory.memory,
                  'shell_lf': self.shell_lf_memory.memory,
                  'lf': self.lf_memory.memory,
                  'leaf_syn_cat': self.leaf_syncat_memory.memory}

        distributions_fpath = join(fpath,'distributions.json')
        with open(distributions_fpath,'w') as f:
            json.dump(to_dump,f)

        self.lf_word_counts.to_pickle(join(fpath,'lf_word_counts.pkl'))
        self.shell_lf_lf_counts.to_pickle(join(fpath,'shell_lf_lf_counts.pkl'))
        self.sem_shell_lf_counts.to_pickle(join(fpath,'sem_shell_lf_counts.pkl'))
        self.sem_word_counts.to_pickle(join(fpath,'sem_word_counts.pkl'))
        self.marginal_syn_counts.to_pickle(join(fpath,'marginal_syn_counts.pkl'))
        self.syn_word_probs.to_pickle(join(fpath,'syn_word_probs.pkl'))

    def show_word_meanings(self,word): # prob of meaning given word assuming flat prior over meanings
        distr = self.wordl.inverse_distribution(word)
        probs = sorted(distr.items(), key=lambda x:x[1])[-15:]
        print(f'\nLearned Meaning for \'{word}\'')
        print('\n'.join([f'{prob:.3f}: {word}' for word,prob in probs]))

    def show_word(self,word,f=None):
        meanings = self.word_to_lf_probs.loc[word].sort_values()[-10:]
        file_print(f'\nLearned meanings for \'{word}\'',f)
        file_print('\n'.join([f'{word}: {100*prob:.2f}%' for word,prob in meanings.items() if prob > 1e-4]),f)

    def get_lf(self,lf_str):
        if lf_str in self.full_lfs_cache:
            lf = self.full_lfs_cache[lf_str]
            #lf.set_cats_as_root(lf_str) # in case was in cache as embedded S, so got X
        else:
            lf = LogicalForm(lf_str,caches=self.caches,parent='START',dblfs=ARGS.dblfs,dbsss=ARGS.dbsss)
            self.full_lfs_cache[lf_str] = lf
        return lf

    def make_parse_node(self,lf_str,words):
        lf = self.get_lf(lf_str)
        #if lf.sem_cats == set('X'):
            #raise RootSemCatError(lf.lf_str)

        if ' '.join([lf_str]+words) in self.parse_node_cache:
            parse_root = self.parse_node_cache[' '.join([lf_str]+words)]
        else:
            parse_root = ParseNode(lf,words,'ROOT')
            for sc in parse_root.sem_cats:
                self.root_sem_cat_memory.observe(sc,1)
            self.parse_node_cache[' '.join([lf_str]+words)] = parse_root
        return parse_root

    def train_one_step(self,lf_strs,words,apply_buffers):
        prob_cache = {}
        root_prob = 0
        new_problem_list = []
        for lfs in lf_strs:
            try:
                root = self.make_parse_node(lfs,words) # words is a list
                new_prob_cache = {}
                new_root_prob = root.propagate_below_probs(self.syntaxl,self.shmeaningl,
                           self.meaningl,self.wordl,new_prob_cache,split_prob=1,is_map=False)
                if lfs == ARGS.dbr:
                    breakpoint()
            except CCGLearnerError as e:
                new_problem_list.append((dpoint,e))
                print(lfs, e)
                continue
            if root.possible_splits == [] and not ARGS.suppress_prints:
                print(words, lfs)
                print('no splits :(')
            if new_root_prob==0:
                print('zero root prob for', lfs, words)
            root.propagate_above_probs(1)
            if words == ARGS.dbsent.split():
                breakpoint()
            for n,p in new_prob_cache.items():
                if n in prob_cache.keys():
                    prob_cache[n] += p
                else:
                    prob_cache[n] = p
            root_prob += new_root_prob
        new_syntaxl_buffer = []
        new_shmeaningl_buffer = []
        new_meaningl_buffer = []
        new_wordl_buffer = []
        vp_prob = 0
        v_prob = 0
        vps = []; vs=[]
        h=lambda l: sum(x.prob for x in l)
        for node, _ in prob_cache.items():
            prob = node.prob
            if node.lf == '':
                breakpoint()
            if node.parent is not None and not node.is_g:
                update_weight = node.prob / root_prob # for conditional
                if node.is_fwd:
                    new_syntaxl_buffer += [(f'{sync} + {ssync}', psync, update_weight) for sync in node.syn_cats for ssync in node.sibling.syn_cats for psync in node.parent.syn_cats]
                else:
                    new_syntaxl_buffer += [(f'{ssync} + {sync}', psync, update_weight) for sync in node.syn_cats for ssync in node.sibling.syn_cats for psync in node.parent.syn_cats]
            leaf_prob = node.above_prob*node.stored_prob_as_leaf/root_prob
            for sync in node.syn_cats:
                self.leaf_syncat_memory.observe(sync, leaf_prob)
            if not leaf_prob > 0:
                print(node)
            lf = node.lf.subtree_string(alpha_normalized=True,recompute=True)
            word_str, lf, shell_lf, sem_cats, syn_cats = node.info_if_leaf()
            if 'S|NP' in sem_cats:
                vps.append(node)
                vp_prob += node.prob
                assert ( np.allclose(h(vps),vp_prob))
            if 'S|NP|NP' in sem_cats:
                vs.append(node)
                v_prob += node.prob
                assert ( np.allclose(h(vs),v_prob))
            for sc in sem_cats:
                self.sem_cat_memory.observe(sc, node.prob)
            for sync in sem_cats:
                self.syn_cat_memory.observe(sync, node.prob)
            self.shell_lf_memory.observe(shell_lf, node.prob)
            self.lf_memory.observe(lf, prob)
            new_syntaxl_buffer += [('leaf',sync,leaf_prob) for sync in syn_cats]
            if any(x[1]=='(N|N)' for x in new_syntaxl_buffer):
                breakpoint()
            if ARGS.condition_on_syncats:
                new_shmeaningl_buffer += [(shell_lf,sync,leaf_prob) for sync in syn_cats]
            else:
                new_shmeaningl_buffer += [(shell_lf,sc,leaf_prob) for sc in sem_cats]
            new_meaningl_buffer.append((lf,shell_lf,leaf_prob))
            new_wordl_buffer.append((word_str,lf,leaf_prob))
        #print(f'VP prob update: {vp_prob}\t V prob update: {v_prob}')
        #v_children=lambda n: [s for ps in n.possible_splits for s in (ps['left'],ps['right']) if 'S|NP|NP' in s.sem_cats]
        #g=lambda n: sum([s.prob for ps in n.possible_splits for s in (ps['left'],ps['right']) if 'S|NP|NP' in s.sem_cats])
        #g=lambda n: h(v_children(n))
        bad_prob = sum(x[2] for x in new_shmeaningl_buffer if x[0]=='lambda $0.lambda $1.vconst $0 $1')
        good_prob = sum(x[2] for x in new_shmeaningl_buffer if x[0]=='lambda $0.lambda $1.vconst $1 $0')
        self.syntaxl.buffers.append(new_syntaxl_buffer)
        self.shmeaningl.buffers.append(new_shmeaningl_buffer)
        self.meaningl.buffers.append(new_meaningl_buffer)
        self.wordl.buffers.append(new_wordl_buffer)
        if apply_buffers:
            self.syntaxl.flush_top_buffer()
            self.shmeaningl.flush_top_buffer()
            self.meaningl.flush_top_buffer()
            self.wordl.flush_top_buffer()
            assert all(len(x.buffers) == ARGS.n_distractors for x in (self.syntaxl,self.shmeaningl,self.meaningl,self.wordl))
        else:
            assert all(len(x.buffers) <= ARGS.n_distractors for x in (self.syntaxl,self.shmeaningl,self.meaningl,self.wordl))
        if '' in self.lf_vocab:
            breakpoint()
        return new_problem_list

    def as_leaf(self, node):
        node.prob_as_leaf(self.syntaxl, self.shmeaningl, self.meaningl, self.wordl, True)

    def probs_of_word_orders(self, ignore_prior):
        if ignore_prior:
            def syn_prob_func(y,x):
                try:
                    return self.syntaxl.memory[x].get(y,0)/self.syntaxl.memory[x]['COUNT']
                except KeyError:
                    return 0
            def shm_prob_func(y,x):
                try:
                    return self.shmeaningl.memory[x].get(y,0)/self.shmeaningl.memory[x]['COUNT']
                except KeyError:
                    return 0
        else:
            syn_prob_func = self.syntaxl.prob
            shm_prob_func = self.shmeaningl.prob
        svo_ = syn_prob_func('NP + S\\NP', 'S')*syn_prob_func('S\\NP/NP + NP', 'S\\NP')
        sov_or_osv = syn_prob_func('NP + S\\NP', 'S')*syn_prob_func('NP + S\\NP\\NP','S\\NP')
        vso_or_vos = syn_prob_func('S/NP + NP', 'S')*syn_prob_func('S/NP/NP + NP','S/NP')
        # it will never have seen this split because disallowed during training
        # so only option is to include prior
        ovs_ = syn_prob_func('S/NP + NP', 'S')*syn_prob_func('NP + S/NP\\NP', 'S/NP')
        if ARGS.condition_on_syncats:
            unnormed_probs = pd.Series({
            'sov': sov_or_osv*(shm_prob_func('lambda $0.lambda $1.vconst $1 $0','S\\NP\\NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $1 $0)','S\\NP\\NP')),
            'svo': svo_*(shm_prob_func('lambda $0.lambda $1.vconst $1 $0','S\\NP/NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $1 $0)','S\\NP/NP')),
            'vso': vso_or_vos*(shm_prob_func('lambda $0.lambda $1.vconst $0 $1','S/NP/NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $0 $1)','S/NP/NP')),
            'vos': vso_or_vos*(shm_prob_func('lambda $0.lambda $1.vconst $1 $0','S/NP/NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $1 $0)','S/NP/NP')),
            'osv': sov_or_osv*(shm_prob_func('lambda $0.lambda $1.vconst $0 $1','S\\NP\\NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $0 $1)','S\\NP\\NP')),
            'ovs': ovs_*(shm_prob_func('lambda $0.lambda $1.vconst $1 $0','S/NP\\NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $1 $0)','S/NP\\NP'))
            }) + 1e-8
        else:
            comb_obj_first = shm_prob_func('lambda $0.lambda $1.vconst $1 $0', 'S|NP|NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $1 $0)', 'S|NP|NP')
            comb_subj_first = shm_prob_func('lambda $0.lambda $1.vconst $0 $1', 'S|NP|NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $0 $1)', 'S|NP|NP')
            unnormed_probs = pd.Series({
            'sov': sov_or_osv*comb_obj_first,
            'svo': svo_*comb_obj_first,
            'vso': vso_or_vos*comb_subj_first,
            'vos': vso_or_vos*comb_obj_first,
            'osv': sov_or_osv*comb_subj_first,
            'ovs': ovs_*comb_obj_first,
            }) + 1e-8

        return unnormed_probs/unnormed_probs.sum()

    def generate_words(self,syn_cat):
        while True:
            split = self.syntaxl.conditional_sample(syn_cat)
            if split == 'leaf':
                break
            if all([s in self.syntaxl.memory for s in split.split(' + ') ]):
                break
        sem_cat = re.sub(r'[\\/]','|',syn_cat)
        if split=='leaf':
            shell_lf = self.shmeaningl.conditional_sample(sem_cat)
            lf = self.meaningl.conditional_sample(shell_lf)
            return self.wordl.conditional_sample(lf)
        else:
            f,g = split.split(' + ')
            return f'{self.generate_words(f)} {self.generate_words(g)}'

    def compute_inverse_probs(self): # prob of meaning given word assuming flat prior over meanings
        self.lf_word_counts =  pd.DataFrame({lf:{w:self.wordl.memory[lf].get(w,0) for w in self.vocab} for lf in self.lf_vocab})
        self.marginal_word_probs = self.lf_word_counts.sum(axis=1)
        self.shell_lf_lf_counts = pd.DataFrame({lfs:{lf:self.meaningl.memory[lfs].get(lf,0) for lf in self.lf_vocab} for lfs in self.shell_lf_vocab})
        self.sem_shell_lf_counts = pd.DataFrame({sync:{lfs:self.shmeaningl.memory[sync.replace('\\','|').replace('/','|')].get(lfs,0) for lfs in self.shell_lf_vocab} for sync in self.sem_cat_vocab})
        self.sem_word_counts = self.lf_word_counts.dot(self.shell_lf_lf_counts).dot(self.sem_shell_lf_counts)
        # transforming syncats to semcats when taking p(shell_lf|syn_cat),
        # implicitly assumes that p(sc|sync) is 1 if compatible and 0# otherwise,
        # so when computing p(sync|sc), we can just restrict to compatible
        # syncats, and ignore p(sc|sync) in Bayes, using only the prior p(sync)
        self.marginal_syn_counts = pd.Series({sync:self.leaf_syncat_memory.prob(sync) for sync in self.syn_cat_vocab})
        self.syn_word_probs = pd.DataFrame({sync:self.sem_word_counts[sc]*self.leaf_syncat_memory.prob(sync) for sc in self.sem_word_counts.columns for sync in possible_syn_cats(sc)})

    def prune_beam(self, full):
        full = sorted(full,key=lambda x:x['prob'])
        beam = []
        counts = {'lf':{},'shell_lf':{},'sem_cat':{},'syn_cat':{}}
        for option in reversed(full):
            if any(k in option.keys() and v.get(option[k],0)==15 for k,v in counts.items()):
                continue
            beam.append(option)
            for key in counts.keys():
                if key in option.keys():
                    counts[key][option[key]] = counts[key].get(option[key],0) + 1
            if len(beam) == self.beam_size:
                break
        return list(reversed(beam))

    def leaf_probs_of_word_span(self,words):
        if words not in self.mwe_vocab and ' ' not in words:
            print(f'\'{words}\' not seen before as a leaf')
        beam = [{'lf':lf,'prob': self.wordl.prob(words,lf)*self.lf_memory.prob(lf)} for lf in self.lf_vocab]
        beam = self.prune_beam(beam)

        beam = [dict(b,shell_lf=shell_lf,prob=b['prob']*self.meaningl.prob(b['lf'],shell_lf)/self.meaningl.marg_prob(b['lf'])*self.shmeaningl.marg_prob(shell_lf))
            for shell_lf in self.shell_lf_vocab for b in beam]
        beam = self.prune_beam(beam)
        beam = [dict(b,sem_cat=sem_cat,prob=b['prob']*self.shmeaningl.prob(b['shell_lf'],sem_cat)/self.shmeaningl.marg_prob(b['shell_lf'])*self.syntaxl.marg_prob(sem_cat))
            for sem_cat in self.sem_cat_vocab for b in beam if sem_cat!='X']
        beam = self.prune_beam(beam)
        beam = [dict(b,prob=b['prob']*self.syntaxl.prob('leaf',b['sem_cat'])) for b in beam]
        beam = self.prune_beam(beam)
        return [dict(b,words=words,rule='leaf',backpointer=None) for b in beam]

    def parse(self,words):
        N = len(words)
        probs_table = np.empty((N,N),dtype='object') #(i,j) will be a dict of len(beam_size) saying probs of top syn_cats for span of len i beginning at index j
        for i in range(N):
            probs_table[0,i] = self.leaf_probs_of_word_span(words[i])

        def add_prob_of_span(i,j):
            word_span = ' '.join(words[j:j+i])
            possible_nexts = self.leaf_probs_of_word_span(word_span)
            for k in range(1,i):
                left_chunk_probs = probs_table[k-1,j]
                right_chunk_probs = probs_table[i-k-1,j+k] # total len is always i, -1s bc 0-index
                print(j,i)
                for left_idx, left_option in enumerate(left_chunk_probs):
                    for right_idx, right_option in enumerate(right_chunk_probs):
                        assert left_option['words'] + ' ' + right_option['words'] == word_span
                        l_cat, r_cat = left_option['sem_cat'], right_option['sem_cat']
                        combined,rule = get_combination(l_cat,r_cat)
                        if combined is None:
                            continue
                        direction,comb_type = rule.split('_')
                        if direction == 'fwd':
                            f,g = left_option['lf'], right_option['lf']
                        elif direction == 'bck':
                            f,g = right_option['lf'], left_option['lf']
                        else:
                            breakpoint()
                        if comb_type == 'cmp':
                            f = logical_type_raise(f)
                        lf = combine_lfs(f,g,comb_type)
                        for csync in possible_syn_cats(combined):
                            lsync, rsync = infer_slash(left_option['sem_cat'],right_option['sem_cat'],csync,rule)
                            split = lsync + ' + ' + rsync
                            mem_correction = self.syn_cat_memory.prob(csync)/self.sem_cat_memory.prob(left_option['sem_cat'])/self.sem_cat_memory.prob(right_option['sem_cat'])
                            prob = left_option['prob']*right_option['prob']*self.syntaxl.prob(split,csync)*mem_correction
                            bckpntr = (k,j,left_idx), (i-k,j+k,right_idx)
                            #backpointer contains the coords in probs_table, and the idx in the
                            #beam, of the two locations that the current one could be split into
                            pn = {'sem_cat':combined,'lf':lf,'backpointer':bckpntr,'rule':rule,'prob':prob,'words':word_span,'syn_cat':csync}
                            possible_nexts.append(pn)

            to_add = self.prune_beam(possible_nexts)
            probs_table[i-1,j] = to_add

        for a in range(2,N+1):
            for b in range(N-a+1):
                add_prob_of_span(a,b)

        for pn in probs_table[N-1,0]:
            if pn is not None:
                pn['syn_cat'] = pn['sem_cat']
                pn['prob'] = pn['prob']*self.root_sem_cat_memory.prob(pn['syn_cat'])

        if len(probs_table[N-1,0]) == 0:
            return 'No parse found'

        def show_parse(num):
            syntax_tree_level = [dict(probs_table[N-1,0][num],idx='1',hor_pos=0, parent='ROOT')]
            all_syntax_tree_levels = []
            for i in range(N):
                new_syntax_tree_level = []
                for item in syntax_tree_level:
                    if item['rule'] == 'leaf':
                        continue
                    backpointer = item['backpointer']
                    (left_len,left_pos,left_idx), (right_len,right_pos,right_idx) = backpointer
                    left_split = probs_table[left_len-1,left_pos][left_idx]
                    right_split = probs_table[right_len-1,right_pos][right_idx]
                    lsync, rsync = infer_slash(left_split['sem_cat'],right_split['sem_cat'],item['syn_cat'],item['rule'])
                    left_split = dict(left_split,idx=item['idx'][:-1] + '01',syn_cat=lsync, parent=item)
                                        #hor_pos=left_pos+(left_len-1)/2 - (N-1)/2)
                    right_split = dict(right_split,idx=item['idx'][:-1] + '21', syn_cat=rsync, parent=item)
                                        #hor_pos=right_pos+(right_len-1)/2 - (N-1)/2)
                    assert 'syn_cat' in left_split.keys() and 'syn_cat' in right_split.keys()
                    item['left_child'] = left_split
                    item['right_child'] = right_split
                    new_syntax_tree_level.append(left_split)
                    new_syntax_tree_level.append(right_split)
                all_syntax_tree_levels.append(syntax_tree_level)
                if all([x['rule']=='leaf' for x in syntax_tree_level]):
                    break
                syntax_tree_level = copy(new_syntax_tree_level)
            if ARGS.db_parse:
                print(probs_table)
                pprint(all_syntax_tree_levels)
            return all_syntax_tree_levels

        favourite_all_syntax_tree_levels = show_parse(-1)
        def _simple_draw_graph(x, depth):
            texttree = '\t'*depth
            for k,v in x.items():
                if k in ('syn_cat', 'lf', 'words'):
                    texttree += f'{k}: {v}\t'
                if k == 'prob':
                    texttree += f'{k}: {v:.4f}\t'
            assert(non_leaf:='left_child' in x.keys()) == ('right_child' in x.keys())
            if non_leaf:
                texttree += '\n'
                texttree += _simple_draw_graph(x['left_child'], depth+1) + '\n'
                texttree += _simple_draw_graph(x['right_child'], depth+1) + '\n'
            return texttree

        print(_simple_draw_graph(favourite_all_syntax_tree_levels[0][0], depth=0))
        self.draw_graph(favourite_all_syntax_tree_levels)

        if ARGS.db_parse:
            breakpoint()
        return [x['sem_cat'] for p in probs_table.flatten() if p is not None for x in p]

    def draw_graph(self,all_syntax_tree_levels,is_gt=False):
        leaves = [n for level in all_syntax_tree_levels for n in level if n['rule']=='leaf']
        leaves.sort(key=lambda x:x['idx'])
        for i,leaf in enumerate(leaves):
            leaf['hor_pos'] = i - len(leaves)/2
        for stl in reversed(all_syntax_tree_levels):
            for n in stl:
                if n['rule']=='leaf':
                    assert 'hor_pos' in n.keys()
                    continue
                elif n['parent'] == 'ROOT':
                    n['hor_pos'] = 0
                else:
                    n['hor_pos'] = (n['left_child']['hor_pos'] + n['right_child']['hor_pos']) / 2
        G=nx.Graph()

        for j,syntax_tree_level in enumerate(all_syntax_tree_levels):
            for i,node in enumerate(syntax_tree_level):
                hor_pos = node['hor_pos']
                if node['rule'] == 'leaf':
                    combined = 'leaf'
                else:
                    combined = node['left_child']['syn_cat']+' + '+node['right_child']['syn_cat']
                split_prob = self.syntaxl.prob(combined,node['syn_cat'])
                if any([x[1]['pos'] ==(node['hor_pos'],-j) for x in G.nodes(data=True)]):
                    breakpoint()
                G.add_node(node['idx'],pos=(hor_pos,-j),label=f"{node['syn_cat']}\n{split_prob:.3f}")
                if node['idx'] != '1': # root
                    G.add_edge(node['idx'],node['idx'][:-2]+'1')

        G.add_node('root',pos=(-1,0),label='ROOT')
        root_weight = self.root_sem_cat_memory.prob(all_syntax_tree_levels[0][0]['syn_cat'])
        G.add_edge('root','1',weight=round(root_weight,3))
        treesize = len(G.nodes())
        n_leaves = len(leaves)
        posses_so_far = nx.get_node_attributes(G,'pos')
        for i,node in enumerate(leaves):
            hor_pos = posses_so_far[node['idx']][0]
            condensed_shell_lf = node['shell_lf'].replace('lambda ','L')
            G.add_node(treesize+i,pos=(hor_pos,-n_leaves),label=condensed_shell_lf)
            shell_lf_prob = self.shmeaningl.prob(node['shell_lf'],node['sem_cat'])
            G.add_edge(treesize+i,node['idx'],weight=round(shell_lf_prob,3))

            condensed_lf = node['lf'].replace('lambda ','L')
            G.add_node(n_leaves+treesize+i, pos=(node['hor_pos'],-n_leaves-1), label=condensed_lf)
            lf_prob = self.meaningl.prob(node['lf'], node['shell_lf'])
            G.add_edge(n_leaves+treesize+i, treesize+i, weight=round(lf_prob,3))

            G.add_node(2*n_leaves+treesize+i, pos=(node['hor_pos'],-n_leaves-2), label=node['words'])
            word_prob = self.wordl.prob(node['words'],node['lf'])
            G.add_edge(2*n_leaves+treesize+i, n_leaves+treesize+i, weight=round(word_prob,3))

        fname = '_'.join([l['words'].replace(' ','_') for l in leaves])
        edge_labels = {k:round(v,2) for k,v in nx.get_edge_attributes(G,'weight').items()}
        node_labels = nx.get_node_attributes(G,'label')
        pos = nx.get_node_attributes(G,'pos')
        nx.draw(G, pos, labels=node_labels, node_color='pink')
        edge_labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,rotate=False)
        if is_gt:
            fname = fname + 'GT'
            plt.title('GT')
        else:
            plt.title('PRED')
        check_dir(graph_dir:=f'experiments/{ARGS.expname}/plotted_graphs')
        plt.savefig(f'{graph_dir}/{fname}.png')
        plt.clf()
        if ARGS.show_graphs:
            os.system(f'/usr/bin/xdg-open "experiments/{ARGS.expname}/plotted_graphs/{fname}.png"')

def remove_vowels(w):
    for v in ('a','e','i','o','u'):
        w = w.replace(v, '')
    return w

if __name__ == "__main__":
    ARGS = argparse.ArgumentParser()
    ARGS.add_argument("--cat_to_sample_from", type=str, default='S')
    ARGS.add_argument("--condition_on_syncats", action="store_true")
    ARGS.add_argument("--db_after", action="store_true")
    ARGS.add_argument("--db_at", type=int, default=-1)
    ARGS.add_argument("--db_parse", action="store_true")
    ARGS.add_argument("--db_prob_changes_above", type=float, default=1.)
    ARGS.add_argument("--dblfs", type=str)
    ARGS.add_argument("--dbr", type=str)
    ARGS.add_argument("--dbsent", type=str, default='')
    ARGS.add_argument("--dbsss", type=str)
    ARGS.add_argument("--eval_alpha", type=float, default=1.0)
    ARGS.add_argument("--exclude_copulae", action="store_true")
    ARGS.add_argument("--exclude_points", type=int, nargs='+', default=[])
    ARGS.add_argument("--expname", type=str,default='tmp')
    ARGS.add_argument("--expname_for_plot_titles", type=str)
    ARGS.add_argument("--jreload_from", type=str)
    ARGS.add_argument("--lr", type=float, default=1.0)
    ARGS.add_argument("--max_lf_len", type=int, default=6)
    ARGS.add_argument("--n_distractors", type=int, default=0)
    ARGS.add_argument("--n_dpoints", type=int, default=-1)
    ARGS.add_argument("--n_epochs", type=int, default=1)
    ARGS.add_argument("--n_generate", type=int, default=0)
    ARGS.add_argument("--n_test", type=int,default=5)
    ARGS.add_argument("--overwrite", action="store_true")
    ARGS.add_argument("--reload_from", type=str)
    ARGS.add_argument("--remove_vowels", action="store_true")
    ARGS.add_argument("--show_graphs", action="store_true")
    ARGS.add_argument("--show_plots", action="store_true")
    ARGS.add_argument("--show_splits", action="store_true")
    ARGS.add_argument("--shuffle", action="store_true")
    ARGS.add_argument("--start_from", type=int, default=0)
    ARGS.add_argument("--suppress_prints", action="store_true")
    ARGS.add_argument("--test_frac", type=float, default=0.1)
    ARGS.add_argument("--test_gts", action="store_true")
    ARGS.add_argument("-d", "--dset", type=str, default='adam')
    ARGS.add_argument("-t","--is_test", action="store_true")
    ARGS.add_argument("-tt","--is_short_test", action="store_true")
    ARGS = ARGS.parse_args()

    ARGS.is_test = ARGS.is_test or ARGS.is_short_test
    if ARGS.dset.lower() not in ARGS.expname.lower():
        ARGS.expname += ARGS.dset[0]

    expdir = f'experiments/{ARGS.dset}'
    if ARGS.is_test:
        ARGS.expname = 'tmp'
        if ARGS.n_dpoints == -1:
            ARGS.n_dpoints = 10
    elif ARGS.expname is None:
        expname = f'{ARGS.n_epochs}_{ARGS.max_lf_len}'
    else:
        expname = ARGS.expname
    if ARGS.jreload_from:
        ARGS.reload_from = ARGS.jreload_from
        #ARGS.db_after = True

    set_experiment_dir(f'experiments/{ARGS.expname}',overwrite=ARGS.overwrite,name_of_trials='experiments/tmp')
    with open(f'data/{ARGS.dset}.json') as f: d=json.load(f)

    if ARGS.shuffle: np.random.shuffle(d['data'])
    NDPS = len(d['data']) if ARGS.n_dpoints == -1 else ARGS.n_dpoints
    all_data = [x for x in d['data'] if len(x['lf'].split()) - x['lf'].count('BARE') <= ARGS.max_lf_len]
    data_to_use = all_data[ARGS.start_from:ARGS.start_from+NDPS]
    train_data = [] if ARGS.jreload_from is not None else data_to_use[:int(len(data_to_use)*(1-ARGS.test_frac))]
    test_data = train_data if ARGS.is_test else data_to_use[-len(train_data):]
    vt = 1 if ARGS.n_dpoints==-1 else ARGS.n_dpoints/len(all_data)
    la = LanguageAcquirer(ARGS.lr, vt)
    if ARGS.reload_from is not None:
        la.load_from(f'experiments/{ARGS.reload_from}')
    start_time = time()
    all_word_order_probs = []
    all_word_order_probs_no_prior = []
    all_prob_changes = []
    problem_list = []
    good_point_idxs = []
    plateaus = []
    gpi = 0
    for epoch_num in range(ARGS.n_epochs):
        epoch_start_time = time()
        for i,dpoint in enumerate(train_data):
            if i in ARGS.exclude_points:
                continue
            words = dpoint['words']
            #if words==['ʔābaʔ', 'ʔat', 'rocā']:
                #continue
            if ARGS.remove_vowels:
                words = [remove_vowels(w) for w in words]
            if dpoint != all_data[i+ARGS.start_from]:
                breakpoint()
            if words[-1] == '?':
                words = words[:-1]
            if i == ARGS.db_at:
                breakpoint()
            start = max(0, i-(ARGS.n_distractors//2))
            stop = min(len(train_data)-1, i+((ARGS.n_distractors+1)//2)+1)
            lf_strs_incl_distractors = [x['lf'] for x in train_data[start:stop]]
            if not ARGS.suppress_prints:
                print(f'{i}th dpoint: {words}, {lf_strs_incl_distractors}')
            apply_buffers = i>=ARGS.n_distractors
            n_words_seen = sum(w in la.vocab for w in words)
            frac_words_seen = n_words_seen/len(words)
            lf = dpoint['lf']
            has_copula = 'v|hasproperty' in lf or 'v|equals' in lf
            #if 'him' in words:
                #breakpoint()
            if not (has_copula and ARGS.exclude_copulae):
               problem_list += la.train_one_step(lf_strs_incl_distractors,words,apply_buffers)
            else:
                print('excluding')
            #if ((i+1)%plot_every == 0 or ARGS.is_test):
            la.eval()
            new_probs_with_prior = la.probs_of_word_orders(False)
            new_probs_without_prior = la.probs_of_word_orders(True)
            if i>0:
                prob_change = ((new_probs_with_prior-all_word_order_probs[-1])**2).sum()
                diff_probs = new_probs_with_prior-all_word_order_probs[-1]
                forder = diff_probs.idxmax()
                good_point = diff_probs['svo'] == diff_probs.max()
                if not good_point and prob_change > 1e-5 and not ARGS.suppress_prints:
                    print(new_probs_with_prior)
                #all_prob_changes.append((prob_change,(words,lf_strs_incl_distractors)))
                if n_words_seen==len(words) and len(split_respecting_brackets(lf))>=3 and not lf.startswith('Q ') and not 'adv|' in lf and len(split_respecting_brackets(lf))>=3 and 'you' not in lf.split() and not lf.startswith('prep|') and not 'not' in lf and not (has_copula and ARGS.dset=='hagar'):
                    gpi+=1
                    if prob_change==0:
                        plateaus.append(dict(dpoint,words=' '.join(dpoint['words'])))
                    else:
                        print(i, round(diff_probs['svo'],7), words, lf)
                all_prob_changes.append({'prob update': prob_change, 'dpoint index': i, 'words':' '.join(words), 'lf':lf, 'good':good_point, 'favoured order':forder,'nseen':n_words_seen})
            good_point_idxs.append(gpi)
            all_word_order_probs.append(new_probs_with_prior)
            all_word_order_probs_no_prior.append(la.probs_of_word_orders(True))
            if i>0 and prob_change>ARGS.db_prob_changes_above:
                breakpoint()
            la.train()
        time_per_dpoint = (time()-epoch_start_time)/len(d['data'])
        print(f'Time per dpoint: {time_per_dpoint:.6f}')
        final_parses = {}
        n_correct_parses = 0

    with open(f'experiments/{ARGS.expname}/failed_dpoints.txt','w') as f:
        for pf,e in problem_list:
            f.write(' '.join(pf['words']) + pf['lf'] + str(e))
    with open(f'experiments/{ARGS.expname}/prob_updates.txt','w') as f:
        for x in sorted(all_prob_changes, key=lambda x:x['prob update'], reverse=True):
            f.write('\t'.join(f'{k}: {" ".join(v) if isinstance(v,list) else round(v,5) if isinstance(v,float) else v}' for k,v in x.items() if k!='good') + '\n')
    with open(f'experiments/{ARGS.expname}/bad_prob_updates.txt','w') as f:
        for x in sorted(all_prob_changes, key=lambda x:x['prob update'], reverse=True):
            if not x['good']:
                f.write('\t'.join(f'{k}: {" ".join(v) if isinstance(v,list) else v}' for k,v in x.items() if k not in ['good', 'forder']) + '\n')
    inverse_probs_start_time = time()
    if len(train_data) > 0:
        la.compute_inverse_probs()
    print(f'Time to compute inverse probs: {time()-inverse_probs_start_time:.3f}s')
    if ARGS.jreload_from is not None:
        df_prior = pd.read_csv(f'experiments/{ARGS.jreload_from}/{ARGS.jreload_from}_word_order_probs.csv', index_col=0)
        df_no_prior = pd.read_csv(f'experiments/{ARGS.jreload_from}/{ARGS.jreload_from}_word_order_probs_no_prior.csv', index_col=0)
    else:
        la.save_to(f'experiments/{ARGS.expname}')
        print(df_prior:=pd.DataFrame(all_word_order_probs))
        df_no_prior=pd.DataFrame(all_word_order_probs_no_prior)
        df_prior.to_csv(f'experiments/{ARGS.expname}/{ARGS.expname}_word_order_probs.csv')
        df_no_prior.to_csv(f'experiments/{ARGS.expname}/{ARGS.expname}_word_order_probs_no_prior.csv')
    cs = ['r','g','b','y','orange','brown']

    def plot_df(df, info='', against_good_point_idxs=False):
        if info != '':
            info = f' {info}'
        xticks = good_point_idxs if against_good_point_idxs else np.arange(len(df))
        for i,c in enumerate(df.columns):
            plt.plot(xticks, df[c], label=c, color=cs[i])
        plt.legend(loc='upper right')
        plt.xlabel('Num Training Points')
        plt.ylabel('Relative Probability')
        plt.title(f'Word Order Probs{info}')
        check_dir('plotted_figures')
        fpath = f'experiments/{ARGS.expname}/word_order_probs{info.replace(" ","_").lower()}.png'
        plt.savefig(fpath)
        if ARGS.show_plots:
            os.system(f'/usr/bin/xdg-open {fpath}')
        plt.clf()

    if ARGS.expname_for_plot_titles is not None:
        dn_info = ARGS.expname_for_plot_titles
    else:
        dn = ARGS.dset[0].upper() + ARGS.dset[1:]
        dn_info = dn if ARGS.n_distractors==0 else f'{dn} {ARGS.n_distractors} distractors'
        dn_info = f'{dn_info} {ARGS.expname[:-1]}'.replace('_',' ')
    if ARGS.jreload_from is None:
        plot_df(df_prior, dn_info)
        plot_df(df_prior, dn_info + ' vs. good points', True)
        plot_df(df_no_prior, f'{ARGS.expname} No Prior')
    print(la.syntaxl.memory.get('S\\NP',None))
    if len(train_data)>=2:
        df = pd.DataFrame(all_prob_changes)
        df = df.sort_values('prob update', ascending=False)
        bad = df.loc[~df['good']].drop('good', axis=1)
        df = df.drop('good', axis=1)
        df.to_csv(f'experiments/{ARGS.expname}/{ARGS.expname}_prob_updates.csv')
        bad.to_csv(f'experiments/{ARGS.expname}/{ARGS.expname}_bad_prob_updates.csv')

    gt_lexicon = all_gt_lexicons[ARGS.dset]
    results_dict = {}
    for w,gts in gt_lexicon:
        pred_lf = la.lf_word_counts.loc[w].idxmax() if w in la.lf_word_counts.index else 'notseen'
        pred_syn = max([s for s in la.syn_word_probs.columns if 'X' not in s], key=lambda x: la.syn_word_probs.loc[w][x] if w in la.syn_word_probs.index else 0)
        lf_correct = any(lf_acc(pred_lf,g.split(' || ')[0]) for g in gts)
        syn_correct = any(pred_syn==g.split(' || ')[1] for g in gts)
        both_correct = any(lf_acc(pred_lf,g.split(' || ')[0]) and pred_syn==g.split(' || ')[1] for g in gts)
        results_dict[w] = {'pred LF':pred_lf, 'pred syncat':pred_syn, 'LF correct':lf_correct, 'syncat correct':syn_correct, 'both correct': both_correct}
    results = pd.DataFrame(results_dict).T
    if not ARGS.suppress_prints:
        print(results)
    word2lf_acc = results['LF correct'].mean()
    word2syn_acc = results['syncat correct'].mean()
    word2both_acc = results['both correct'].mean()
    if ARGS.db_after:
        breakpoint()
    results.to_csv(f'experiments/{ARGS.expname}/{ARGS.expname}_full_preds_and_scores.csv')
    with open(f'experiments/{ARGS.expname}/{ARGS.expname}_summary.txt','w') as f:
        if ARGS.n_generate > 0:
            file_print(f'\nSamples for type {ARGS.cat_to_sample_from}:',f)
        for _ in range(ARGS.n_generate):
            generated = la.generate_words(ARGS.cat_to_sample_from)
            if any([x['words'][:-1]==generated.split() for x in d['data']]):
                file_print(f'{generated}: seen during training',f)
            else:
                file_print(f'{generated}: not seen during training',f)
        file_print(f'Total run time: {time()-start_time:.3f}s',f)
        file_print(f'Word to LF accuracy: {word2lf_acc:.4f}',f)
        file_print(f'Word to syncat accuracy: {word2syn_acc:.4f}',f)
        file_print(f'Full lexical accuracy: {word2both_acc:.4f}',f)
        if len(train_data) == 0:
            file_print('Trainset was empty, no word_order_probs logged',f)
        else:
            file_print('Final word order probs:',f)
            for k,v in all_word_order_probs[-1].items():
                file_print(f'{k}: {v:.6f}',f)
    #la.parse('do you like it'.split())
    la.parse('you see him'.split())
    la.parse('did he see it'.split())
    la.parse('the pencil dropped the name'.split())
    la.parse('did the pencil see the name'.split())
    considered_sem_cats = []
    for dpoint in test_data[:ARGS.n_test]:
        considered_sem_cats += la.parse(dpoint['words'])
    if ARGS.test_gts:
        for sent,gt in gts.items():
            la.parse(gt[0][0]['words'].split())
            la.draw_graph(gt,is_gt=True)
    pdf = pd.DataFrame(plateaus)
    pprint(set(considered_sem_cats))
    breakpoint()
    print(pdf)
    pdf.to_csv(f'experiments/{ARGS.expname}/{ARGS.expname}_plateaus.csv')
