import numpy as np
from dl_utils.misc import check_dir, set_experiment_dir
import os
import networkx as nx
import matplotlib.pyplot as plt
from copy import copy
from os.path import join
import pandas as pd
from utils import get_combination, combine_lfs, possible_syncs, lf_cat_congruent, split_respecting_brackets, base_cats_from_str, non_directional, cat_components, is_type_raised, is_atomic, maybe_debrac, IWFF
#from is_wff import IWFF
from errors import CCGLearnerError
from time import time
import argparse
from abc import ABC
import re
from parser import LogicalForm, ParseNode
import json


reslashes = re.compile(r'[\\/\|]')
reshellsplitpoints = re.compile(r'[ \(\)\.]')

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

def comb_either_way_around(y, x):
    if y == 'leaf':
        return True
    outcat, _ = get_combination(*y.split(' + '))
    if outcat is None: #only time this should happen is during inference with crossed cmp
        return False
#    if not is_direct_congruent(outcat,x):
#        other_way_around, _ = get_combination(y.split(' + ')[1], y.split(' + ')[0])
#        if not is_direct_congruent(other_way_around, x):
#            return False
    return True

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

    def topk(self, k=10):
        return sorted([(k,self.prob(k)) for k in self.memory.keys() if k not in ('X','COUNT')],key=lambda x:x[1],reverse=True)[:k]

    def sample(self):
        options,unnormed_probs = zip(*[z for z in self.memory.items() if z[0]!='COUNT'])
        probs = np.array(unnormed_probs)/sum(unnormed_probs)
        return np.random.choice(options,p=probs)

class BaseDirichletProcessLearner(ABC):
    def __init__(self,alpha):
        self.alpha = alpha
        self.base_alpha = alpha
        self.memory = {}
        self.base_distribution_cache = {}
        self.marg_prob_cache = {}
        self.prob_cache = {}
        self.buffers = []
        self.long_term_buffer = []
        self.is_training = True
        self.use_cache = True

    def train(self):
        self.alpha = self.base_alpha
        self.is_training = True

    def eval(self):
        self.alpha = 1
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
        prob = (mx+self.alpha*base_prob)/(self.memory[x]['COUNT']+self.alpha)
        return prob

    def marg_prob(self, x, ignore_cache=False):
        if ignore_cache:
            return self._marg_prob(x)
        elif x in self.marg_prob_cache:
            return self.marg_prob_cache[x]
        else:
            prob = self._marg_prob(x)
            self.marg_prob_cache[x] = prob
            return prob

    def _marg_prob(self, x):
        numerator = sum(self.prob(x, k)*v['COUNT'] for k,v in self.memory.items())
        denominator = sum(v['COUNT'] for k,v in self.memory.items())
        prob = numerator / denominator
        return prob

    def prob(self, y, x):
        if (y,x) in self.prob_cache and self.use_cache:
            return self.prob_cache[(y,x)]
        prob = self._prob(y,x)
        if self.use_cache:
            self.prob_cache[(y,x)] = prob
        return prob

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

    def refresh(self):
        self.memory = {}
        self.flush_selected_buffer('long-term')
        self.long_term_buffer = []

    def flush_selected_buffer(self, bname):
        if bname=='top':
            selected_buffer = self.buffers.pop(0)
        elif bname=='long-term':
            selected_buffer = self.long_term_buffer
        else:
            breakpoint()
        self.flush_buffer(selected_buffer)

    def flush_buffer(self, buffer):
        for _ in range(len(buffer)):
            y, x, weight = buffer.pop()
            self.observe(y, x, weight)
        assert len(buffer) == 0
        self.prob_cache = {}

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

    def topk(self,x,k=10):
        return sorted(self.memory[x].items(), key=lambda x:x[1], reverse=True)[:k]

class CCGDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        n_slashes = len(reslashes.findall(x))
        #return (0.5555)*0.9**(n_slashes+1) # Omri 2017 had 0.2
        return 0.9**(n_slashes+1) # Omri 2017 had 0.2

    def observe(self,y,x,weight):
        if comb_either_way_around(y, x):
            self._observe(y, x, weight)

    #@override
    def _marg_prob(self, x):
        denominator = sum(v['COUNT'] for v in self.memory.values())
        base_prob = self.base_distribution(x)
        numerator = sum(v['COUNT'] for k,v in self.memory.items() if k in possible_syncs(x))
        return (numerator + self.alpha*base_prob) / (denominator + self.alpha)

    def prob(self,y,x, can_be_weird_svo=False):
        if (y,x) in self.prob_cache:
            return self.prob_cache[(y,x)]
        assert x != 'NP + S/NP\\NP' or not self.is_training
        if not comb_either_way_around(y, x):
            prob = 0
        else:
            prob = self._prob(y,x)
        self.prob_cache[(y,x)] = prob
        return prob

class ShellMeaningDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        parts = reshellsplitpoints.split( x)
        parts = [p for p in parts if p!='' and '$' not in p]
        return 4**-len(parts)
        #return 4**-len(parts)/3

    def prob(self,y,x):
        if (y,x) in self.prob_cache:
            return self.prob_cache[(y,x)]
        if not lf_cat_congruent(y,x):
            prob = 0
        else:
            prob = self._prob(y,x)
        self.prob_cache[(y,x)] = prob
        return prob

class MeaningDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        parts = reshellsplitpoints.split(x)
        parts = [p for p in parts if p!='' and '$' not in p]
        return 4**-len(parts)
        #return 4**-len(parts)/3

class WordSpanDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution(self,x):
        x = x.replace(' ','').replace('\'','')
        assert len(x) > 0
        if ARGS.remove_vowels:
            return 14**-len(x)
        else:
            return 14**-len(x) # kinda approximate num phonetic chunks

class LanguageAcquirer():
    def __init__(self, lr, vocab_thresh):
        self.lr = lr
        self.syntaxl = CCGDirichletProcessLearner(100)
        self.shmeaningl = ShellMeaningDirichletProcessLearner(1)
        self.meaningl = MeaningDirichletProcessLearner(1)
        self.wordl = WordSpanDirichletProcessLearner(0.25)
        self.full_lfs_cache = {} # maps lf_strings to LogicalForm objects
        self.caches = {'splits':{}, 'cats':{}}
        self.parse_node_cache = {} # maps utterances (str) to ParseNode objects, including splits
        self.root_sem_cat_memory = DirichletProcess(1) # counts of the sem_cats it's seen as roots
        self.leaf_syncat_memory = DirichletProcess(1)
        self.vocab_thresh = vocab_thresh
        self.beam_size = 10
        self.learners = {'syntax': self.syntaxl, 'shmeaning': self.shmeaningl, 'meaning':self.meaningl, 'word': self.wordl}
        self.iwff = IWFF()
        self.is_training = True

    def disable_caching(self):
        for l in self.learners.values():
            l.use_cache = False

    def enable_caching(self):
        for l in self.learners.values():
            l.use_cache = True

    def refresh(self):
        for b in self.buffers.values():
            b.refresh
        self.root_sem_cat_memory.memory = {}
        self.leaf_syncat_memory.memory = {}

    def train(self):
        self.is_training = True
        self.syntaxl.train()
        self.shmeaningl.train()
        self.meaningl.train()
        self.wordl.train()

    def eval(self):
        self.is_training = False
        self.syntaxl.eval()
        self.shmeaningl.eval()
        self.meaningl.eval()
        self.wordl.eval()

    @property
    def full_lf_vocab(self):
        return [k for k,v in self.wordl.memory.items()]

    @property
    def full_shell_lf_vocab(self):
        return [k for k,v in self.meaningl.memory.items()]

    @property
    def full_syn_cat_vocab(self):
        return [k for k,v in self.shmeaningl.memory.items()]

    @property
    def full_sync_vocab(self):
        return [k for k,v in self.syntaxl.memory.items()]

    @property
    def full_splits_vocab(self):
        return list(set([k for x in self.syntaxl.memory.values() for k in x.keys() if k!='COUNT']))

    @property
    def full_mwe_vocab(self):
        return list(set([k for x in self.wordl.memory.values() for k in x.keys() if k!='COUNT']))

    @property
    def lf_vocab(self):
        return [k for k,v in self.wordl.memory.items() if v['COUNT']>self.vocab_thresh]

    @property
    def shell_lf_vocab(self):
        return [k for k,v in self.meaningl.memory.items() if v['COUNT']>self.vocab_thresh]

    @property
    def syn_cat_vocab(self):
        return [k for k,v in self.shmeaningl.memory.items() if v['COUNT']>self.vocab_thresh]

    @property
    def sync_vocab(self):
        return [k for k,v in self.syntaxl.memory.items() if v['COUNT']>self.vocab_thresh]

    @property
    def splits_vocab(self):
        return list(set([k for x in self.syntaxl.memory.values() for k,v in x.items() if k!='COUNT' if v>self.vocab_thresh]))

    @property
    def mwe_vocab(self):
        return list(set([k for x in self.wordl.memory.values() for k,v in x.items() if k!='COUNT' if v>self.vocab_thresh]))

    @property
    def vocab(self):
        return list(set([w for x in self.wordl.memory.values() for w in x.keys() if ' ' not in w]))

    def load_from(self, fpath, full_counts=False):
        distributions_fpath = join(fpath,'distributions.json')
        with open(distributions_fpath) as f:
            to_load = json.load(f)
        self.syntaxl.set_from_dict(to_load['syntax'])
        self.shmeaningl.set_from_dict(to_load['shmeaningl'])
        self.meaningl.set_from_dict(to_load['meaning'])
        self.wordl.set_from_dict(to_load['word'])
        self.root_sem_cat_memory.memory = to_load['root_sem_cat']
        self.leaf_syncat_memory.memory = to_load['leaf_sync']

        if full_counts:
            self.lf_word_counts = pd.read_pickle(join(fpath,'lf_word_counts.pkl'))
            self.shell_lf_lf_counts = pd.read_pickle(join(fpath,'shell_lf_lf_counts.pkl'))
            self.sem_shell_lf_counts = pd.read_pickle(join(fpath,'sem_shell_lf_counts.pkl'))
            self.sem_word_counts = pd.read_pickle(join(fpath,'sem_word_counts.pkl'))
            self.syn_word_probs = pd.read_pickle(join(fpath,'syn_word_probs.pkl'))

    def save_to(self, fpath, full_counts=False):
        check_dir(fpath)
        to_dump = {'syntax': {'memory':self.syntaxl.memory,
                  'base_distribution_cache':self.syntaxl.base_distribution_cache},
                  'shmeaningl': {'memory':self.shmeaningl.memory,
                  'base_distribution_cache':self.shmeaningl.base_distribution_cache},
                  'meaning': {'memory':self.meaningl.memory,
                  'base_distribution_cache':self.meaningl.base_distribution_cache},
                  'word': {'memory':self.wordl.memory,
                  'base_distribution_cache':self.wordl.base_distribution_cache},
                  'root_sem_cat': self.root_sem_cat_memory.memory,
                  'leaf_sync': self.leaf_syncat_memory.memory}

        distributions_fpath = join(fpath,'distributions.json')
        with open(distributions_fpath,'w') as f:
            json.dump(to_dump,f)

        if full_counts:
            self.lf_word_counts.to_pickle(join(fpath,'lf_word_counts.pkl'))
            self.shell_lf_lf_counts.to_pickle(join(fpath,'shell_lf_lf_counts.pkl'))
            self.sem_shell_lf_counts.to_pickle(join(fpath,'sem_shell_lf_counts.pkl'))
            self.sem_word_counts.to_pickle(join(fpath,'sem_word_counts.pkl'))
            self.syn_word_probs.to_pickle(join(fpath,'syn_word_probs.pkl'))

    def get_lf(self, lf_str, words):
        if lf_str in self.full_lfs_cache:
            lf = self.full_lfs_cache[lf_str + '||' + ' '.join(words)]
            #lf.set_cats_as_root(lf_str) # in case was in cache as embedded S, so got X
        else:
            #if words[0] in ('what', 'who', 'how', 'where', 'when', 'which'):
            if len(words)==2 and words[0]=='which':
                specified_cats = {'Swhq|(Sq|NP)'}
            if any(w in words for w in ('what', 'who', 'how', 'where', 'when', 'which')):
                specified_cats = {'Swhq'}# if 'Q' in lf_str else {'Swh'}
            elif lf_str.startswith('Q'):
                specified_cats = {'Sq'}
            elif 'lambda' in lf_str:
                specified_cats = {'VP'}
            elif len(split_respecting_brackets(lf_str))==2 and lf_str.startswith('prep'):
                specified_cats = {'S|NP|(S|NP)'}
            elif len(lf_str.split())==2 and lf_str.split()[1].startswith('n|') and not lf_str.startswith('v'):
                specified_cats = {'NP'}
            else:
                specified_cats = {'S'}
            lf = LogicalForm(lf_str,caches=self.caches,parent='START',dblfs=ARGS.dblfs,dbsss=ARGS.dbsss, verbose_as=ARGS.verbose_as, specified_cats=specified_cats)
            self.full_lfs_cache[lf_str + '||' + ' '.join(words)] = lf
        lf.infer_splits()
        return lf

    def make_parse_node(self, lf_str, words):
        lf = self.get_lf(lf_str, words)

        if ' '.join([lf_str]+words) in self.parse_node_cache:
            parse_root = self.parse_node_cache[' '.join([lf_str]+words)]
        else:
            if len(lf.sem_cats) > 1:
                if lf.node_type != 'lmbda':
                    lf.sem_cats = [s for s in lf.sem_cats if '|' not in s]
            if len(lf.sem_cats) > 1:
                breakpoint()
            lfsc = list(lf.sem_cats)[0]
            parse_root = ParseNode(lf,words,'ROOT', sync=lfsc)
            for sc in parse_root.sem_cats:
                self.root_sem_cat_memory.observe(sc,1)
            self.parse_node_cache[' '.join([lf_str]+words)] = parse_root
        return parse_root

    def step(self, lf_strs, words, apply_buffers, put_in_ltbs, mode, print_train_interps, is_nonce_test=False):
        all_descs = []
        root_prob = 0
        best_lf_prob = 0
        best_lf = 'none'
        new_problem_list = []
        for lfs in lf_strs:
            try:
                root = self.make_parse_node(lfs,words) # words is a list
                new_root_prob = root.propagate_below_probs(self.syntaxl,self.shmeaningl,
                           self.meaningl,self.wordl,split_prob=1,is_map=False)
                if print_train_interps or 'prog' in lfs:
                    _, leaf_cats = self.test_with_gt(lfs, words)
                    print(leaf_cats)
                #if ARGS.print_gtparsestrs:
                    #gtparsestr, is_good, is_root_good = root.gt_parse()
                    #print(gtparsestr)
                if lfs == ARGS.dbr or (ARGS.dbw is not None and ARGS.dbw in ' '.join(words)):
                    if not ARGS.print_train_interps:
                        self.eval()
                        self.disable_caching()
                        self.test_with_gt(lfs, words)
                        self.enable_caching()
                        self.train()
                    gtparsestr, is_good, is_root_good = root.gt_parse()
                    print(gtparsestr)
                    self.test_with_gt_graphical(lfs, words, 9, False, show_graph=False)
                    breakpoint()
            except CCGLearnerError as e:
                new_problem_list.append((dpoint,e))
                if mode=='train':
                    print(lfs, e)
                continue
            if root.splits == [] and not ARGS.suppress_prints and mode=='train':
                print('no splits :(')
            if new_root_prob==0 and mode=='train':
                print('zero root prob for', lfs, words)
            root.propagate_above_probs(1)
            if words == ARGS.dbsent.split():
                breakpoint()
            root_prob += new_root_prob
            all_descs += root.descs
            if new_root_prob > best_lf_prob:
                best_lf_prob = new_root_prob
                best_lf = lfs

        if mode=='test':
            return best_lf
        buffers = {x:[] for x in self.learners.keys()}
        for node in all_descs:
            if node.parent is not None and not node.is_g:
                update_weight = node.prob / root_prob # for conditional
                if node.is_fwd:
                    buffers['syntax'].append((f'{node.sync} + {node.sibling.sync}',node.parent.sync, update_weight))
                else:
                    buffers['syntax'].append((f'{node.sibling.sync} + {node.sync}',node.parent.sync, update_weight))
            leaf_prob = node.above_prob*node.stored_prob_as_leaf/root_prob
            self.leaf_syncat_memory.observe(node.sync, leaf_prob)
            if not leaf_prob > 0:
                print(node)
            lf = node.lf.subtree_string(alpha_normalized=True,recompute=True)
            word_str, lf, shell_lf, sem_cats, sync = node.info_if_leaf()
            buffers['syntax'].append(('leaf',sync,leaf_prob))
            if ARGS.no_condition_on_syncats:
                buffers['shmeaning'] += [(shell_lf,sc,leaf_prob) for sc in sem_cats]
            else:
                buffers['shmeaning'].append((shell_lf,sync,leaf_prob))
            buffers['meaning'].append((lf,shell_lf,leaf_prob))
            buffers['word'].append((word_str,lf,leaf_prob))

        for lname, learner in self.learners.items():
            buffer = buffers[lname]
            if is_nonce_test:
                learner.buffers = [buffer] + learner.buffers
            else:
                learner.buffers.append(buffer)
            if put_in_ltbs:
                learner.long_term_buffer += buffer
            if apply_buffers:
                learner.flush_selected_buffer('top')
                #assert len(learner.buffers) == ARGS.n_distractors # no longer works if using step in nonce experiment
            else:
                assert len(learner.buffers) <= ARGS.n_distractors
        #print(self.syntaxl.memory.get('S', None))
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
            shm_prob_func = self.shmeaningl.prob
        svo_ = self.syntaxl.marg_prob('S\\NP/NP', ignore_cache=True)
        sov_or_osv = self.syntaxl.marg_prob('S\\NP\\NP', ignore_cache=True)
        vso_or_vos = self.syntaxl.marg_prob('S/NP/NP', ignore_cache=True)
        ovs_ = self.syntaxl.marg_prob('S/NP\\NP', ignore_cache=True)
        if ARGS.no_condition_on_syncats:
            comb_obj_first = shm_prob_func('lambda $0.lambda $1.vconst $0 $1', 'S|NP|NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $1 $0)', 'S|NP|NP')
            comb_subj_first = shm_prob_func('lambda $0.lambda $1.vconst $1 $0', 'S|NP|NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $0 $1)', 'S|NP|NP')
            unnormed_probs = pd.Series({
            'sov': sov_or_osv*comb_obj_first,
            'svo': svo_*comb_obj_first,
            'vso': vso_or_vos*comb_subj_first,
            'vos': vso_or_vos*comb_obj_first,
            'osv': sov_or_osv*comb_subj_first,
            'ovs': ovs_*comb_obj_first,
            }) + 1e-8
        else:
            unnormed_probs = pd.Series({
            'sov': sov_or_osv*(shm_prob_func('lambda $0.lambda $1.vconst $0 $1','S/NP/NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $1 $0)','S/NP/NP')),
            'svo': svo_*(shm_prob_func('lambda $0.lambda $1.vconst $0 $1','S\\NP/NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $1 $0)','S\\NP/NP')),
            'vso': vso_or_vos*(shm_prob_func('lambda $0.lambda $1.vconst $1 $0','S\\NP\\NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $0 $1)','S\\NP\\NP')),
            'vos': vso_or_vos*(shm_prob_func('lambda $0.lambda $1.vconst $0 $1','S\\NP\\NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $1 $0)','S\\NP\\NP')),
            'osv': sov_or_osv*(shm_prob_func('lambda $0.lambda $1.vconst $1 $0','S\\NP\\NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $0 $1)','S\\NP\\NP')),
            'ovs': ovs_*(shm_prob_func('lambda $0.lambda $1.vconst $0 $1','S/NP\\NP')+shm_prob_func('lambda $0.lambda $1.neg (vconst $1 $0)','S/NP\\NP'))
            }) + 1e-8

        return unnormed_probs/unnormed_probs.sum()

    def generate_words(self, sync=None, depth=0):
        if sync is None:
            while sync in (None, 'Swh'):
                sync = self.root_sem_cat_memory.sample()
        while True:
            split = self.syntaxl.conditional_sample(sync)
            if split == 'leaf':
                break
            if all([s in self.syntaxl.memory for s in split.split(' + ') ]):
                break
        if split=='leaf':
            if ARGS.no_condition_on_syncats:
                sem_cat = re.sub(r'[\\/]','|',sync)
                shell_lf = self.shmeaningl.conditional_sample(sem_cat)
            else:
                shell_lf = self.shmeaningl.conditional_sample(sync)
            lf = self.meaningl.conditional_sample(shell_lf)
            words = self.wordl.conditional_sample(lf)
            #print(' '*depth + f'lf: {lf}  sync: {sync} words: {words}')
            return words
        else:
            #print(' '*depth + f'sync: {sync}')
            f,g = split.split(' + ')
            return f'{self.generate_words(f, depth+1)} {self.generate_words(g, depth+1)}'

    def compute_inverse_probs(self): # prob of meaning given word assuming flat prior over meanings
        self.lf_word_counts =  pd.DataFrame({lf:{w:self.wordl.memory[lf].get(w,0) for w in self.vocab} for lf in self.lf_vocab})
        self.marginal_word_probs = self.lf_word_counts.sum(axis=1)
        self.shell_lf_lf_counts = pd.DataFrame({lfs:{lf:self.meaningl.memory[lfs].get(lf,0) for lf in self.lf_vocab} for lfs in self.shell_lf_vocab})
        x_for_lfs = self.syn_cat_vocab if ARGS.no_condition_on_syncats else self.full_sync_vocab
        #self.sem_shell_lf_counts = pd.DataFrame({sync:{lfs:self.shmeaningl.memory[sync.replace('\\','|').replace('/','|')].get(lfs,0) for lfs in self.shell_lf_vocab} for sync in x_for_lfs})
        self.sem_shell_lf_counts = pd.DataFrame({sync:{lfs:self.shmeaningl.memory[sync].get(lfs,0) for lfs in self.shell_lf_vocab} for sync in x_for_lfs})
        self.sem_word_counts = self.lf_word_counts.dot(self.shell_lf_lf_counts).dot(self.sem_shell_lf_counts)
        # transforming syncats to semcats when taking p(shell_lf|sync),
        # implicitly assumes that p(sc|sync) is 1 if compatible and 0# otherwise,
        # so when computing p(sync|sc), we can just restrict to compatible
        # syncats, and ignore p(sc|sync) in Bayes, using only the prior p(sync)
        #self.marginal_syn_counts = pd.Series({sync:self.leaf_syncat_memory.prob(sync) for sync in self.sync_vocab})
        self.marginal_syn_counts = pd.Series({sync:self.leaf_syncat_memory.prob(sync) for sync in self.sync_vocab})
        self.syn_word_probs = pd.DataFrame({sync:self.sem_word_counts[sc]*self.leaf_syncat_memory.prob(sync) for sc in self.sem_word_counts.columns for sync in possible_syncs(sc)})

    def prune_beam(self, full):
        full = sorted(full,key=lambda x:x['prob'])
        beam = []
        counts = {'lf':{},'shell_lf':{},'sem_cat':{},'sync':{}}
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
        #if words not in self.mwe_vocab and ' ' not in words:
            #print(f'\'{words}\' not seen before as a leaf')
        if words == ARGS.db_word_parse:
            breakpoint()
        beam = [{'lf':lf,'prob': self.wordl.prob(words,lf)*self.meaningl.marg_prob(lf)} for lf in self.lf_vocab if self.iwff.is_wellformed_lf(lf, should_be_normed=True)]
        beam = self.prune_beam(beam)

        beam = [dict(b,shell_lf=shell_lf,prob=b['prob']*self.meaningl.prob(b['lf'],shell_lf)/self.meaningl.marg_prob(b['lf'])*self.shmeaningl.marg_prob(shell_lf))
            for shell_lf in self.shell_lf_vocab for b in beam]
        beam = self.prune_beam(beam)
        # shouldn't I be conditioning on syncats here? working fine for now, but look into later
        beam = [dict(b,sync=syn_cat,prob=b['prob']*self.shmeaningl.prob(b['shell_lf'],syn_cat)/self.shmeaningl.marg_prob(b['shell_lf'])*self.syntaxl.marg_prob(syn_cat))
            for syn_cat in self.syn_cat_vocab for b in beam if syn_cat!='X']
        beam = [b for b in beam if lf_basecat_congruent(b['lf'], non_directional(b['sync']))]
        beam = self.prune_beam(beam)
        #beam = [dict(b, sync=ps) for b in beam for ps in possible_syncs(b['syn_cat'])]
        #beam = self.prune_beam(beam)
        beam = [dict(b,prob=b['prob']*self.syntaxl.prob('leaf',b['sync'])) for b in beam]
        beam = self.prune_beam(beam)
        if any(x['lf'] == 'Q (mod|do-past pro:per|you) n|name' for x in beam):
            breakpoint()
        return [dict(b,words=words,rule='leaf',backpointer=None) for b in beam]

    def parse(self, words, fs=10, show_graph=False):
        N = len(words)
        probs_table = np.empty((N,N),dtype='object') #(i,j) will be a dict of len(beam_size) saying probs of top syncs for span of len i beginning at index j
        for i in range(N):
            probs_table[0,i] = self.leaf_probs_of_word_span(words[i])

        def add_prob_of_span(i,j):
            word_span = ' '.join(words[j:j+i])
            possible_nexts = self.leaf_probs_of_word_span(word_span)
            if any(x['lf'] == 'Q (mod|do-past pro:per|you) n|name' for x in possible_nexts):
                breakpoint()
            for k in range(1,i):
                left_chunk_probs = probs_table[k-1,j]
                right_chunk_probs = probs_table[i-k-1,j+k] # total len is always i, -1s bc 0-index
                for left_idx, left_option in enumerate(left_chunk_probs):
                    for right_idx, right_option in enumerate(right_chunk_probs):
                        assert left_option['words'] + ' ' + right_option['words'] == word_span
                        l_cat, r_cat = left_option['sync'], right_option['sync']
                        #combined,rule = get_combination(l_cat, r_cat, enforce_q_match=False)
                        combined,rule = get_combination(l_cat, r_cat)
                        if combined is None:
                            continue
                        direction,comb_type = rule.split('_')
                        if direction == 'fwd':
                            f,g = left_option['lf'], right_option['lf']
                        elif direction == 'bck':
                            f,g = right_option['lf'], left_option['lf']
                        else:
                            breakpoint()
                        if comb_type == 'cmp' and (not 'lambda' in f or not 'lambda' in g): # won't be able to cmp then so must be a mistake
                            continue
                        lf = combine_lfs(f,g,comb_type)
                        if not self.iwff.is_wellformed_lf(lf, should_be_normed=True):
                            continue
                        what_cats_should_be, _ = base_cats_from_str(lf)
                        if 'X' not in what_cats_should_be and combined not in what_cats_should_be:
                            continue
                        if not lf_cat_congruent(lf, combined):
                            continue
                        #if left_option['lf'] == 'lambda $0.Q (mod|can ($0 pro:per|you))' and left_option['sync'] == 'Sq/VP' and left_option['words'] == 'can you' and right_option['lf'] == 'lambda $0.lambda $1.v|see $0 $1' and right_option['sync']=='VP/NP' and right_option['words']=='see':
                            #breakpoint()
                        lsync, rsync = left_option['sync'], right_option['sync']
                        split = lsync + ' + ' + rsync
                        mem_correction_sync = self.syntaxl.marg_prob(combined)/self.syntaxl.marg_prob(left_option['sync'])/self.syntaxl.marg_prob(right_option['sync'])
                        prob = left_option['prob']*right_option['prob']*self.syntaxl.prob(split,combined)*mem_correction_sync
                        bckpntr = (k,j,left_idx), (i-k,j+k,right_idx)
                        #backpointer contains the coords in probs_table (except x is +1), and the idx in the
                        #beam, of the two locations that the current one could be split into
                        pn = {'lf':lf,'backpointer':bckpntr,'rule':rule,'prob':prob,'words':word_span,'sync':combined}
                        possible_nexts.append(pn)
            to_add = []
            for lf, sync in set((x['lf'], x['sync']) for x in possible_nexts):
                matching_examples = [x for x in possible_nexts if x['lf']==lf and x['sync']==sync]
                best_example = max(matching_examples, key=lambda x:x['prob'])
                to_add.append(best_example)

            to_add = self.prune_beam(to_add)
            probs_table[i-1,j] = to_add

        for a in range(2,N+1):
            for b in range(N-a+1):
                add_prob_of_span(a,b)

        for pn in probs_table[N-1,0]:
            if pn is not None:
                #pn['sync'] = pn['sem_cat']
                pn['prob'] = pn['prob']*self.root_sem_cat_memory.prob(pn['sync'])

        probs_table[N-1, 0] = self.prune_beam(probs_table[N-1,0])
        if len(probs_table[N-1,0]) == 0:
            return {'lf': 'NOT FOUND', 'sync': 'NOT FOUND'}

        def show_parse(num):
            tree_level = [dict(probs_table[N-1,0][num],idx='1',hor_pos=0, parent='ROOT')]
            all_tree_levels = []
            for i in range(N):
                new_tree_level = []
                for item in tree_level:
                    if item['rule'] == 'leaf':
                        continue
                    backpointer = item['backpointer']
                    (left_len,left_pos,left_idx), (right_len,right_pos,right_idx) = backpointer
                    left_split = probs_table[left_len-1,left_pos][left_idx]
                    right_split = probs_table[right_len-1,right_pos][right_idx]
                    lsync, rsync = left_split['sync'], right_split['sync']
                    left_split = dict(left_split,idx=item['idx'][:-1] + '01',sync=lsync, parent=item)
                    right_split = dict(right_split,idx=item['idx'][:-1] + '21', sync=rsync, parent=item)
                    assert 'sync' in left_split.keys() and 'sync' in right_split.keys()
                    item['left_child'] = left_split
                    item['right_child'] = right_split
                    new_tree_level.append(left_split)
                    new_tree_level.append(right_split)
                all_tree_levels.append(tree_level)
                if all([x['rule']=='leaf' for x in tree_level]):
                    break
                tree_level = copy(new_tree_level)
            return all_tree_levels

        favourite_all_tree_levels = show_parse(-1)
        def _simple_draw_graph(x, depth):
            texttree = '\t'*depth
            for k,v in x.items():
                if k in ('sync', 'lf', 'words'):
                    texttree += f'{k}: {v}\t'
                if k == 'prob':
                    texttree += f'{k}: {v:.4f}\t'
            non_leaf = 'left_child' in x.keys()
            assert non_leaf == ('right_child' in x.keys())
            if non_leaf:
                texttree += '\n'
                texttree += _simple_draw_graph(x['left_child'], depth+1) + '\n'
                texttree += _simple_draw_graph(x['right_child'], depth+1) + '\n'
            return texttree

        assert all(x['rule']=='leaf' for x in favourite_all_tree_levels[-1])
        leaf_cats = [(x['words'], x['sync']) for level in favourite_all_tree_levels for x in level if x['rule']=='leaf']
        if set(' '.join(x[0] for x in leaf_cats).split()) != set(words):
            breakpoint()
        if ARGS.print_test_parses:
            print(_simple_draw_graph(favourite_all_tree_levels[0][0], depth=0))
        self.draw_graph(favourite_all_tree_levels, fs=fs, show_graph=show_graph)

        if ARGS.db_parse:
            breakpoint()
        return probs_table[-1,0][-1], leaf_cats

    def draw_graph(self,all_tree_levels, fs, show_graph, clip=True):
        leaves = [n for level in all_tree_levels for n in level if n['rule']=='leaf']
        leaves.sort(key=lambda x:x['idx'])
        for i,leaf in enumerate(leaves):
            leaf['hor_pos'] = i - len(leaves)/2
        for stl in reversed(all_tree_levels):
            for n in stl:
                if n['rule']=='leaf':
                    assert 'hor_pos' in n.keys()
                    continue
                elif n['parent'] == 'ROOT':
                    n['hor_pos'] = 0
                else:
                    n['hor_pos'] = (n['left_child']['hor_pos'] + n['right_child']['hor_pos']) / 2
        G=nx.Graph()

        for j,tree_level in enumerate(all_tree_levels):
            for i,node in enumerate(tree_level):
                hor_pos = node['hor_pos']
                if node['rule'] == 'leaf':
                    combined = 'leaf'
                else:
                    combined = node['left_child']['sync']+' + '+node['right_child']['sync']
                split_prob = self.syntaxl.prob(combined,node['sync'])
                if any([x[1]['pos'] ==(node['hor_pos'],-j) for x in G.nodes(data=True)]):
                    breakpoint()
                G.add_node(node['idx'],pos=(hor_pos,-j),label=f"{node['sync']}\n{split_prob:.3f}")
                if node['idx'] != '1': # root
                    G.add_edge(node['idx'],node['idx'][:-2]+'1')

        G.add_node('root',pos=(-1,0),label='ROOT')
        root_weight = self.root_sem_cat_memory.prob(all_tree_levels[0][0]['sync'])
        G.add_edge('root','1',weight=round(root_weight,3))
        treesize = len(G.nodes())
        n_leaves = len(leaves)
        posses_so_far = nx.get_node_attributes(G,'pos')
        def condense(s):
            s = s.replace('lambda ','\u03BB').replace('$0', 'x').replace('$1', 'y').replace('$2', 'z')
            #if '|' in s:
                #breakpoint()
            s = re.sub(r'[a-z:]*?\|', '', s)
            s = s.replace('-zero', '')
            return s

        for i,node in enumerate(leaves):
            hor_pos = posses_so_far[node['idx']][0]
            condensed_shell_lf = condense(node['shell_lf'])
            G.add_node(treesize+i,pos=(hor_pos,-n_leaves),label=condensed_shell_lf)
            shell_lf_prob = self.shmeaningl.prob(node['shell_lf'],node['sync'])
            G.add_edge(treesize+i,node['idx'],weight=round(shell_lf_prob,3))

            condensed_lf = condense(node['lf'])
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
        #color_map = ['pink']*(len(G)-len(leaves)) + ['#d3ffce']*len(leaves)
        color_map = ['#ffc0cb' if isinstance(x, str) or x < len(G.nodes) - len(leaves) else '#d3ffce' for x in G.nodes]
        nx.draw(G, pos, labels=node_labels, node_color=color_map, font_size=fs)
        edge_labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, rotate=False, font_size=fs)
        check_dir(graph_dir:=f'experiments/{ARGS.expname}/plotted_graphs')
        plt.axis('off')
        contract_ratio = 1 if n_leaves==1 or (not clip) else 1.1
        axis = plt.gca()
        axis.set_xlim([contract_ratio*x for x in axis.get_xlim()])
        #axis.set_ylim([1.1*y for y in axis.get_ylim()])
        plt.tight_layout()
        plt.savefig(f'{graph_dir}/{fname}.png')
        plt.clf()
        if show_graph:
            os.system(f'/usr/bin/xdg-open "experiments/{ARGS.expname}/plotted_graphs/{fname}.png"')

    def test_with_gt(self, lf, words, print_parse_tree=True):
        root = la.make_parse_node(lf, words)
        root.propagate_below_probs(la.syntaxl,la.shmeaningl,la.meaningl,la.wordl,split_prob=1,is_map=True)
        def _simple_draw_graph(x, depth):
            texttree = '  '*depth
            texttree += f'LF: {x.lf_str}\tSync: {x.sync}\tWords: {" ".join(x.words)}\tRelprob: {x.rel_prob:.4f}'
            left_child, right_child, _ = x.best_split
            non_leaf = left_child!='leaf'
            assert non_leaf == (right_child!='leaf')
            if non_leaf:
                texttree += '\n'
                #texttree += _simple_draw_graph(left_child, depth+1) + '\n'
                #texttree += _simple_draw_graph(right_child, depth+1) + '\n'
                left_texttree, left_leaves = _simple_draw_graph(left_child, depth+1)
                right_texttree, right_leaves = _simple_draw_graph(right_child, depth+1)
                texttree += left_texttree + '\n' + right_texttree + '\n'
                leaves = left_leaves + right_leaves
            else:
                leaves = [(' '.join(x.words), x.sync)]
            return texttree, leaves

        st, leaves = _simple_draw_graph(root, 0)
        if print_parse_tree:
            print(st)
        return st, leaves

    def test_with_gt_graphical(self, lf, words, font_size, clip, show_graph):
        root = la.make_parse_node(lf, words)
        root.propagate_below_probs(la.syntaxl,la.shmeaningl,la.meaningl,la.wordl,split_prob=1,is_map=True)
        def _graph_dict_from_node(x):
            shell_lf = x.lf.subtree_string(as_shell=True, alpha_normalized=True)
            z = {'sync':x.sync, 'words':' '.join(x.words), 'lf':x.lf_str, 'shell_lf':shell_lf}
            z['rule'] = 'leaf' if x.best_split[0]=='leaf' else 'nonleaf'
            return z

        root_graph_dict = dict(_graph_dict_from_node(root), parent='ROOT', idx='1')
        all_tree_levels = [[root_graph_dict]]
        frontier = [(root, root_graph_dict)]
        for i in range(len(root.words)):
            new_frontier = []
            tree_level = []
            for pnode, pnode_dict in frontier:
                if pnode.is_leaf:
                    continue
                left_child, right_child, _ = pnode.best_split
                assert (left_child=='leaf') == (right_child=='leaf')
                if left_child == 'leaf':
                    continue
                left_split = dict(_graph_dict_from_node(left_child), idx=pnode_dict['idx'][:-1] + '01', parent='nonroot')
                right_split = dict(_graph_dict_from_node(right_child), idx=pnode_dict['idx'][:-1] + '21', parent='nonroot')
                assert 'sync' in left_split.keys() and 'sync' in right_split.keys()
                pnode_dict['left_child'] = left_split
                pnode_dict['right_child'] = right_split
                tree_level.append(left_split)
                tree_level.append(right_split)
                new_frontier.append((left_child, left_split)); new_frontier.append((right_child, right_split))
            all_tree_levels.append(tree_level)
            frontier = new_frontier
            if all([x['rule']=='leaf' for x in tree_level]):
                break
        assert all(isinstance(x, dict) for tl in all_tree_levels for x in tl)
        self.draw_graph(all_tree_levels, fs=font_size, clip=clip, show_graph=show_graph)

def remove_vowels(w):
    for v in ('a','e','i','o','u'):
        w = w.replace(v, '')
    return w

def lf_basecat_congruent(lf, semcat):
    basecats, _ = base_cats_from_str(lf)
    if 'X' in basecats:
        return True
    if semcat in basecats:
        return True
    for bsc in basecats:
        if is_type_raised(lf):
            if is_atomic(semcat):
                continue
            outcat, _, incat = cat_components(semcat)
            if is_atomic(incat):
                continue
            outincat, _, inincat = cat_components(maybe_debrac(incat))
            if outincat==outcat and inincat==bsc:
                return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--break-train-at", type=int, default=-1)
    parser.add_argument("--cat-to-sample-from", type=str, default='s')
    parser.add_argument("--no-condition-on-syncats", action="store_true")
    parser.add_argument("--db-after", action="store_true")
    parser.add_argument("--db-at", type=int, default=-1)
    parser.add_argument("--db-before", action="store_true")
    parser.add_argument("--db-parse", action="store_true")
    parser.add_argument("--db-prob-changes-above", type=float, default=1.)
    parser.add_argument("--db-word-parse", type=str, default='')
    parser.add_argument("--dblfs", type=str)
    parser.add_argument("--dbr", type=str)
    parser.add_argument("--dbsent", type=str, default='')
    parser.add_argument("--dbsss", type=str)
    parser.add_argument("--dbw", type=str)
    parser.add_argument("--eval-alpha", type=float, default=1.0)
    parser.add_argument("--exclude-copulae", action="store_true")
    parser.add_argument("--exclude-points", type=int, nargs='+', default=[])
    parser.add_argument("--expname", type=str,default='tmp')
    parser.add_argument("--expname-for-plot-titles", type=str)
    parser.add_argument("--ignore-words", type=str, nargs='+')
    parser.add_argument("--jdblfs", type=str)
    parser.add_argument("--jdbr", type=str)
    parser.add_argument("--jreload-from", type=str)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--max-lf-len", type=int, default=6)
    parser.add_argument("--max-sent-len", type=int, default=6)
    parser.add_argument("--n-distractors", type=int, default=0)
    parser.add_argument("--n-dpoints", type=int, default=-1)
    parser.add_argument("--n-epochs", type=int, default=1)
    parser.add_argument("--n-generate", type=int, default=300)
    parser.add_argument("--n-test", type=int,default=-1)
    parser.add_argument("--no-test-roots", action="store_true")
    parser.add_argument("--no-one-trial", action="store_true")
    parser.add_argument("--no-ltbs", action="store_true")
    parser.add_argument("--nonce", type=str, choices=['jj', 'mb', 'var', 'none'], default='none')
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--print-gtparsestrs", action="store_true")
    parser.add_argument("--print-test-parses", action="store_true")
    parser.add_argument("--print-train-interps", action="store_true")
    parser.add_argument("--print-word-acc", action="store_true")
    parser.add_argument("--reload-from", type=str)
    parser.add_argument("--remove-vowels", action="store_true")
    parser.add_argument("--show-graphs", action="store_true")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--show-splits", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--start-from", type=int,default=0)
    parser.add_argument("--start-test-from", type=int,default=0)
    parser.add_argument("--suppress-prints", action='store_true')
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--test-incr", action="store_true")
    parser.add_argument("--test-next-chunk", action="store_true")
    parser.add_argument("--test-parses", action="store_true")
    parser.add_argument("--verbose-as", action="store_true")
    parser.add_argument("--vary-subj", action="store_true")
    parser.add_argument("--vary-obj-det", action="store_true")
    parser.add_argument("--vary-obj-noun", action="store_true")
    parser.add_argument("--vocab-thresh", type=float, default=0.0)
    parser.add_argument("-d", "--dset", type=str, default='adam')
    parser.add_argument("-t","--is-test", action="store_true")
    parser.add_argument("-tt","--is-short-test", action="store_true")
    ARGS = parser.parse_args()

    if ARGS.is_test:
        ARGS.expname = 'tmp'
        if ARGS.n_dpoints == -1:
            ARGS.n_dpoints = 10
    elif ARGS.expname is None:
        expname = f'{ARGS.n_epochs}_{ARGS.max_lf_len}'
    else:
        expname = ARGS.expname

    set_experiment_dir(f'experiments/{ARGS.expname}',overwrite=ARGS.overwrite,name_of_trials='experiments/tmp')
    with open(f'data/{ARGS.dset}.json') as f: d=json.load(f)

    if ARGS.shuffle: np.random.shuffle(d['data'])
    NDPS = len(d['data']) if ARGS.n_dpoints == -1 else ARGS.n_dpoints
    all_data = [x for x in d['data'] if len(x['lf'].split()) - x['lf'].count('BARE') <= ARGS.max_lf_len and len(x['words'])<=ARGS.max_sent_len]
    if ARGS.ignore_words:
        all_data = [x for x in all_data if not any(w in x['words'] for w in ARGS.ignore_words)]
    data_to_use = all_data[ARGS.start_from:ARGS.start_from+NDPS]
    train_data = data_to_use[:int(len(data_to_use)*(1-ARGS.test_frac))]
    test_data = train_data if ARGS.is_test else data_to_use[int(len(data_to_use)*(1-ARGS.test_frac)):]
    if ARGS.break_train_at == -1:
        ARGS.break_train_at = len(train_data)
    vt = 1 if ARGS.n_dpoints==-1 else ARGS.n_dpoints/len(all_data) # threshold for excluding rare vocab
    la = LanguageAcquirer(ARGS.lr, vt)
    if ARGS.reload_from is not None:
        la.load_from(f'experiments/{ARGS.reload_from}')

    chunk_size = max(1, len(train_data)//20)

    ltb_idxs = [100]
    start_time = time()

    train_start_time = time()
    for i,dpoint in enumerate(train_data):
        if i==ARGS.break_train_at:
            break
        words = dpoint['words']
        if i == ARGS.db_at:
            breakpoint()
        start = max(0, i-(ARGS.n_distractors//2))
        stop = min(len(train_data), i+((ARGS.n_distractors+1)//2)+1)
        lf_strs_incl_distractors = [x['lf'] for x in train_data[start:stop]]
        if not ARGS.suppress_prints:
            print(f'{i}th dpoint: {words}, {lf_strs_incl_distractors}')
        buffers = i>=ARGS.n_distractors
        n_words_seen = sum(w in la.vocab for w in words)
        ltbs = i>= ltb_idxs[0]
        la.step(lf_strs_incl_distractors, words, buffers, ltbs, mode='train', print_train_interps=ARGS.print_train_interps)
        if i in ltb_idxs[1:] and not ARGS.no_ltbs:
            for b in la.learners.values():
                b.refresh()

    time_per_dpoint = (time()-train_start_time)/len(d['data'])
    print(f'Time per dpoint: {time_per_dpoint:.6f}')
    if len(train_data) > 0:
        la.compute_inverse_probs()
    inverse_probs_start_time = time()
    print(f'Time to compute inverse probs: {time()-inverse_probs_start_time:.3f}s')

    la.save_to(f'experiments/{ARGS.expname}')
