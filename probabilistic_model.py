import sys
from tqdm import tqdm
import numpy as np
from dl_utils.misc import check_dir, set_experiment_dir
import os
import networkx as nx
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from os.path import join
import shutil
import pandas as pd
from utils import file_print, get_combination, combine_lfs, possible_syncs, infer_slash, lf_cat_congruent, lf_acc, split_respecting_brackets, base_cats_from_str, non_directional, cat_components, is_type_raised, is_atomic, maybe_debrac, IWFF
#from is_wff import IWFF
from errors import CCGLearnerError
from time import time
import argparse
from abc import ABC
import re
from parser import LogicalForm, ParseNode
import json
from learner_config import all_gt_lexicons
from wh_test_annots import whw_cat_dict


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
        return (0.5555)*0.9**(n_slashes+1) # Omri 2017 had 0.2

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

    def load_from(self,fpath):
        distributions_fpath = join(fpath,'distributions.json')
        with open(distributions_fpath) as f:
            to_load = json.load(f)
        self.syntaxl.set_from_dict(to_load['syntax'])
        self.shmeaningl.set_from_dict(to_load['shmeaningl'])
        self.meaningl.set_from_dict(to_load['meaning'])
        self.wordl.set_from_dict(to_load['word'])
        self.root_sem_cat_memory.memory = to_load['root_sem_cat']
        self.leaf_syncat_memory.memory = to_load['leaf_sync']

        self.lf_word_counts = pd.read_pickle(join(fpath,'lf_word_counts.pkl'))
        self.shell_lf_lf_counts = pd.read_pickle(join(fpath,'shell_lf_lf_counts.pkl'))
        self.sem_shell_lf_counts = pd.read_pickle(join(fpath,'sem_shell_lf_counts.pkl'))
        self.sem_word_counts = pd.read_pickle(join(fpath,'sem_word_counts.pkl'))
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
                  'leaf_sync': self.leaf_syncat_memory.memory}

        distributions_fpath = join(fpath,'distributions.json')
        with open(distributions_fpath,'w') as f:
            json.dump(to_dump,f)

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

    def step(self, lf_strs, words, apply_buffers, put_in_ltbs, mode, print_train_interps):
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
                if print_train_interps or 'what' in words:
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
                    self.test_with_gt_graphical(lfs, words, 9, False)
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
            learner.buffers.append(buffer)
            if put_in_ltbs:
                learner.long_term_buffer += buffer
            if apply_buffers:
                learner.flush_selected_buffer('top')
                assert len(learner.buffers) == ARGS.n_distractors
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
            syn_prob_func = self.syntaxl.prob
            shm_prob_func = self.shmeaningl.prob
        #svo_ = syn_prob_func('NP + S\\NP', 'S')*syn_prob_func('S\\NP/NP + NP', 'S\\NP')
        svo_ = self.syntaxl.marg_prob('S\\NP/NP', ignore_cache=True)
        #sov_or_osv = syn_prob_func('NP + S\\NP', 'S')*syn_prob_func('NP + S\\NP\\NP','S\\NP')
        sov_or_osv = self.syntaxl.marg_prob('S\\NP\\NP', ignore_cache=True)
        #vso_or_vos = syn_prob_func('S/NP + NP', 'S')*syn_prob_func('S/NP/NP + NP','S/NP')
        vso_or_vos = self.syntaxl.marg_prob('S/NP/NP', ignore_cache=True)
        # it will never have seen this split because disallowed during training
        # so only option is to include prior
        #ovs_ = syn_prob_func('S/NP + NP', 'S')*syn_prob_func('NP + S/NP\\NP', 'S/NP')
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

    def parse(self, words, fs=10):
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
        self.draw_graph(favourite_all_tree_levels, fs=fs)

        if ARGS.db_parse:
            breakpoint()
        return probs_table[-1,0][-1], leaf_cats

    def draw_graph(self,all_tree_levels, fs, node_color='#d3ffce', clip=True):
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
        nx.draw(G, pos, labels=node_labels, node_color=node_color, font_size=fs)
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
        if ARGS.show_graphs:
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

    def test_with_gt_graphical(self, lf, words, font_size, clip):
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
        self.draw_graph(all_tree_levels, node_color='pink', fs=font_size, clip=clip)

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
    parser.add_argument("--test-nonce", action="store_true")
    parser.add_argument("--test-parses", action="store_true")
    parser.add_argument("--verbose-as", action="store_true")
    parser.add_argument("--vocab-thresh", type=float, default=0.0)
    parser.add_argument("-d", "--dset", type=str, default='adam')
    parser.add_argument("-t","--is-test", action="store_true")
    parser.add_argument("-tt","--is-short-test", action="store_true")
    ARGS = parser.parse_args()

    ARGS.is_test = ARGS.is_test or ARGS.is_short_test
    if ARGS.dset.lower() not in ARGS.expname.lower():
        ARGS.expname += ARGS.dset[0]

    if ARGS.jdbr is not None:
        assert ARGS.jdblfs is None
        ARGS.dbr = ARGS.jdbr
        just_lf = ARGS.jdbr
    elif ARGS.jdblfs is not None:
        ARGS.dblfs = ARGS.jdblfs
        just_lf = ARGS.jdblfs
    else:
        just_lf = None
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

    set_experiment_dir(f'experiments/{ARGS.expname}',overwrite=ARGS.overwrite,name_of_trials='experiments/tmp')
    with open(f'data/{ARGS.dset}.json') as f: d=json.load(f)

    if ARGS.shuffle: np.random.shuffle(d['data'])
    NDPS = len(d['data']) if ARGS.n_dpoints == -1 else ARGS.n_dpoints
    all_data = [x for x in d['data'] if len(x['lf'].split()) - x['lf'].count('BARE') <= ARGS.max_lf_len and len(x['words'])<=ARGS.max_sent_len]
    if ARGS.ignore_words:
        all_data = [x for x in all_data if not any(w in x['words'] for w in ARGS.ignore_words)]
    data_to_use = all_data[ARGS.start_from:ARGS.start_from+NDPS]
    if just_lf is not None:
        train_data = [x for x in data_to_use if x['lf']==just_lf]
        test_data = []
        if len(train_data)==0:
            sys.exit('no data points match this lf, check for typo')
    else:
        train_data = [] if ARGS.jreload_from is not None else data_to_use[:int(len(data_to_use)*(1-ARGS.test_frac))]
        test_data = train_data if ARGS.is_test else data_to_use[int(len(data_to_use)*(1-ARGS.test_frac)):]
    if ARGS.break_train_at == -1:
        ARGS.break_train_at = len(train_data)
    vt = 1 if ARGS.n_dpoints==-1 else ARGS.n_dpoints/len(all_data)
    la = LanguageAcquirer(ARGS.lr, vt)
    if ARGS.reload_from is not None:
        la.load_from(f'experiments/{ARGS.reload_from}')

    chunk_size = max(1, len(train_data)//20)
    if ARGS.db_before:
        breakpoint()

    nonce_dps = {
        'trans': ('you dax the pencil', ['v|dax pro:per|you (det:art|the n|pencil)', 'v|dax (det:art|the n|pencil) pro:per|you']),
        'negmod': ('you can n\'t dax the pencil', ['not (mod|can (v|dax pro:per|you (det:art|the n|pencil)))', 'not (mod|can (v|dax (det:art|the n|pencil) pro:per|you))']),
        'intrans': ('you dax', ['v|dax pro:per|you', 'lambda $0.v|dax pro:per|you $0']),
        'modal': ('you can dax the pencil', ['mod|can (v|dax pro:per|you (det:art|the n|pencil))', 'mod|can (v|dax (det:art|the n|pencil) pro:per|you)']),
        'polarq': ('will you dax the pencil', ['Q (mod|will (v|dax pro:per|you (det:art|the n|pencil)))', 'Q (mod|will (v|dax (det:art|the n|pencil) pro:per|you))']),
        'negpolarq': ('can n\'t you dax the pencil', ['Q (not (mod|can (v|dax pro:per|you (det:art|the n|pencil))))', 'Q (not (mod|can (v|dax (det:art|the n|pencil) pro:per|you)))']),
        'prog': ('you \'re dax the pencil', ['cop|pres-2s (v|dax pro:per|you (det:art|the n|pencil))', 'cop|pres-2s (v|dax (det:art|the n|pencil) pro:per|you)']),
        'wh': ('who will you dax', ['Q (mod|will (v|dax pro:int|WHO pro:per|you))', 'Q (mod|will (v|dax pro:per|you pro:int|WHO))']),
        'negwh': ('who can n\'t you dax', ['Q (not (mod|can (v|dax pro:int|WHO pro:per|you)))', 'Q (not (mod|can (v|dax pro:per|you pro:int|WHO)))']),
        }

    const_types = ('other', 'intrans', 'trans', 'ditrans', 'mod', 'q', 'whq', 'prog', 'neg', 'imp')
    def test_parses(testset):
        la.eval()
        n_correct = 0
        n_select_correct = 0
        n_with_seen_words = 0
        la.vocab_thresh = 0
        n_test_dists = 4
        acc_by_type = {k:[] for k in const_types}
        for i, dpoint in enumerate(pbar:=tqdm(testset)):
            lf, words = dpoint['lf'], dpoint['words']
            if all(w in la.mwe_vocab for w in words):
                n_with_seen_words += 1
            predicted_parse, leaf_cats = la.parse(words)
            cts = []
            if any(x in lf for x in ['adv|', 'part']):
                cts.append('other')
            if lf.startswith('Q'):
                if 'WH' in lf:
                    cts.append('whq')
                else:
                    cts.append('q')
            if 'prog' in lf:
                cts.append('prog')
                if 'not' in lf:
                    cts.append('neg')
            if 'mod' in lf:
                cts.append('mod')
                if 'not' in lf:
                    cts.append('neg')
            splits = split_respecting_brackets(lf)
            if any(w.startswith('v|') for w in splits):
                if len(splits) == 2:
                    cts.append('intrans')
                elif len(splits) == 3:
                    cts.append('trans')
                elif len(splits) in [4,5]: # 5 if imperative ditrans
                    cts.append('ditrans')
                else:
                    breakpoint()
                    cts.append('other')
                if lf.startswith('lambda $0.'):
                    cts.append('imp')
            if cts==[]:
                cts.append('other')

            if (is_correct:=predicted_parse['lf'] == lf):
                n_correct += 1
            for ct in cts:
                acc_by_type[ct].append(is_correct)
            start = max(0, i-(n_test_dists//2))
            stop = min(len(train_data), i+((n_test_dists+1)//2)+1)
            lf_strs_incl_distractors = [x['lf'] for x in testset[start:stop]]
            selected_lf = la.step(lf_strs_incl_distractors, words, apply_buffers=False, put_in_ltbs=False, mode='test', print_train_interps=ARGS.print_train_interps)
            if selected_lf == lf:
                n_select_correct += 1
            pbar.set_description(f'Acc: {n_correct/(n_with_seen_words+1e-8):.4f}  Harshacc: {n_correct/(i+1):.4f} Selectacc: {n_select_correct/(i+1):.4f}')
        la.vocab_thresh = 1
        if n_with_seen_words == 0:
            return 0,0,0,{k:0 for k in const_types}
        harsh_acc = n_correct/len(test_data)
        select_acc = n_select_correct/len(test_data)
        acc_excl_nws = n_correct/n_with_seen_words
        acc_by_type = {k:np.array(v).mean() for k,v in acc_by_type.items()}
        if acc_excl_nws > 1:
            breakpoint()
        print(f'num test points with only seen words: {n_with_seen_words}')
        print(f'harsh acc: {harsh_acc:.3f}\nacc excluding new words: {acc_excl_nws:.3f}')
        la.train()
        return acc_excl_nws, harsh_acc, select_acc, acc_by_type

    whw_list = ['what', 'who', 'which']
    def test_wh_words(testset):
        n_correct = 0
        denom = 0
        n_correct_wgt = 0
        n_wrong_wgt = 0
        n_wrong = 0
        n_no_parses = 0
        seen_words = []
        gts = []
        gt_words = []
        preds = []
        preds_wgt = []
        for i, dpoint in enumerate(pbar:=tqdm(testset)):
            lf, words = dpoint['lf'], dpoint['words']
            if ' '.join(words) in seen_words:
                continue
            if words[0] in whw_list:
                gt_whw_cat = whw_cat_dict[' '.join(words)]
                predicted_parse, leaf_cats = la.parse(words)
                if leaf_cats=='sync':
                    pred_whw_cat_as_list = []
                    n_no_parses += 1
                    preds.append('no parse')
                else:
                    pred_whw_cat_as_list = [c for w,c in leaf_cats if any(x in w for x in whw_list)] # means 'not found'
                _, leaf_cats_wgt = la.test_with_gt(lf, words, print_parse_tree=False)
                pred_whw_cat_as_list_wgt = [c for w,c in leaf_cats_wgt if any(x in w for x in whw_list)]
                if len(pred_whw_cat_as_list) > 0:
                    pred_whw_cat = pred_whw_cat_as_list[0]
                    preds.append(pred_whw_cat)
                    if pred_whw_cat == gt_whw_cat:
                        n_correct += 1
                    else:
                        n_wrong += 1
                        print(leaf_cats)
                    #if gt_whw_cat != 'other': denom += 1
                if len(pred_whw_cat_as_list_wgt) > 0:
                    pred_whw_cat_wgt = pred_whw_cat_as_list_wgt[0]
                    preds_wgt.append(pred_whw_cat_wgt)
                    if pred_whw_cat_wgt == gt_whw_cat:
                        n_correct_wgt += 1
                    else:
                        print('WITH GT:', leaf_cats_wgt)
                        n_wrong_wgt += 1
                    #if gt_whw_cat != 'other':
                        #denom_wgt += 1
                else:
                    breakpoint()
                denom+=1
                seen_words.append(' '.join(words))
                gts.append(gt_whw_cat)
                gt_words.append(words)
                pbar.set_description(f'Acc: {n_correct/(denom+1e-9):.3f} Acc-wgt: {n_correct_wgt/(denom+1e-9):.3f}')
        if len(preds) != denom:
            breakpoint()
        print(f'ncorrect no gt: {n_correct} ncorrect WITH GT: {n_correct_wgt} n-wrong no gt: {n_wrong} n-wrong WITH GT: {n_wrong_wgt} n no parses: {n_no_parses} denom: {denom}')
        if n_correct_wgt > denom:
            breakpoint()
        return n_correct/(denom+1e-9), n_correct_wgt/(denom+1e-9), preds, preds_wgt, gts, gt_words

    def test_nonce():
        results = {}
        before = deepcopy(la.syntaxl.memory)
        la.save_to('/tmp')
        la.eval()
        la.disable_caching()
        #la.wordl.alpha = 0.05
        for const_type, (sent, lfs) in nonce_dps.items():
            print(const_type)
            la.step(lfs, sent.split(), apply_buffers=True, put_in_ltbs=False, mode='train', print_train_interps=ARGS.print_train_interps)
            daxm = 'lambda $0.v|dax $0' if const_type=='intrans' else 'lambda $0.lambda $1.v|dax $0 $1'
            la.wordl.alpha = 0.1
            #la.meaningl.alpha = 0
            cpp = la.wordl.prob('dax', daxm)# * la.meaningl.marg_prob(daxm)
            #cpp = la.wordl.prob('dax', daxm)*la.shmeaningl.prob('lambda $0.lambda $1.vconst $0 $1','S\\NP/NP')
            la.wordl.alpha = 1
            #la.meaningl.alpha = 1
            #cpp = la.wordl.memory[daxm]['dax'] / la.wordl.memory[daxm]['COUNT']
            assert cpp <= 1
            results[const_type] = cpp
            #print(la.wordl.topk(daxm))
            la.load_from('/tmp')
            #shutil.rmtree.remove_dirs('tmp')
            assert la.syntaxl.memory == before
        la.train()
        la.enable_caching()
        print(results)
        return results

    def test_word_acc():
        la.compute_inverse_probs()
        results_dict = {}
        for w,gts in all_gt_lexicons[ARGS.dset]:
            pred_lf = la.lf_word_counts.loc[w].idxmax() if w in la.lf_word_counts.index else 'notseen'
            pred_syn = max([s for s in la.syn_word_probs.columns if 'X' not in s], key=lambda x: la.syn_word_probs.loc[w][x] if w in la.syn_word_probs.index else 0)
            lf_correct = any(lf_acc(pred_lf,g.split(' || ')[0]) for g in gts)
            syn_correct = any(pred_syn==g.split(' || ')[1] for g in gts)
            both_correct = any(lf_acc(pred_lf,g.split(' || ')[0]) and pred_syn==g.split(' || ')[1] for g in gts)
            results_dict[w] = {'pred LF':pred_lf, 'pred syncat':pred_syn, 'LF correct':lf_correct, 'syncat correct':syn_correct, 'both correct': both_correct}
        results = pd.DataFrame(results_dict).T
        word2lf_acc = results['LF correct'].mean()
        word2syn_acc = results['syncat correct'].mean()
        word2both_acc = results['both correct'].mean()
        return results, word2lf_acc, word2syn_acc, word2both_acc

    #ltb_idxs = [100, 300, 1000]
    ltb_idxs = [100]
    start_time = time()
    all_word_order_probs = []
    gpi = 0
    all_accs = []
    all_harsh_accs = []
    all_select_accs = []
    all_next_accs = []
    all_next_harsh_accs = []
    all_next_select_accs = []
    all_w2lf_accs = []
    all_w2syn_accs = []
    all_w2both_accs = []
    all_whw_accs = []
    all_whwwgt_accs = []
    all_whw_preds = []
    all_whwwgt_preds = []
    all_whq_order_probs = []
    all_whq_order_probs_not_marg = []
    all_nonce_results = {k:[] for k in nonce_dps.keys()}
    all_accs_by_type = {k:[] for k in const_types}
    final_chunk = test_data[ARGS.start_test_from:]
    if ARGS.n_test != -1:
        final_chunk = test_data[:ARGS.n_test]

    train_start_time = time()
    for i,dpoint in enumerate(train_data):
        if i==ARGS.break_train_at:
            break
        words = dpoint['words']
        if ARGS.remove_vowels:
            words = [remove_vowels(w) for w in words]
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
        la.eval()
        whq_order_probs = {
        #'fwdfwd1': la.syntaxl.prob('Swhq/(Sq/NP) + Sq/NP', 'Swhq', ignore_cache=True),
        #'fwdbck1': la.syntaxl.prob('Swhq/(Sq\\NP) + Sq\\NP', 'Swhq', ignore_cache=True),
        'fwdfwd': la.syntaxl.marg_prob('Swhq/(Sq/NP)', ignore_cache=True),
        'fwdbck': la.syntaxl.marg_prob('Swhq/(Sq\\NP)', ignore_cache=True),
        'bckfwd': la.syntaxl.marg_prob('Swhq\\(Sq/NP)', ignore_cache=True),
        'bckbck': la.syntaxl.marg_prob('Swhq\\(Sq\\NP)', ignore_cache=True),
        }
        whq_order_probs_not_marg = {
        #'fwdfwd1': la.syntaxl.prob('Swhq/(Sq/NP) + Sq/NP', 'Swhq', ignore_cache=True),
        #'fwdbck1': la.syntaxl.prob('Swhq/(Sq\\NP) + Sq\\NP', 'Swhq', ignore_cache=True),
        'fwdfwd': la.syntaxl.prob('Swhq/(Sq/NP) + Sq/NP', 'Swhq'),
        'fwdbck': la.syntaxl.prob('Swhq/(Sq\\NP) + Sq\\NP', 'Swhq'),
        'bckfwd': la.syntaxl.prob('Sq/NP + Swhq\\(Sq/NP)', 'Swhq'),
        'bckbck': la.syntaxl.prob('Sq\\NP + Swhq\\(Sq\\NP)', 'Swhq'),
        }
        whq_order_probs = {k:v/sum(whq_order_probs.values()) for k,v in whq_order_probs.items()}
        whq_order_probs_not_marg = {k:v/sum(whq_order_probs_not_marg.values()) for k,v in whq_order_probs_not_marg.items()}
        all_whq_order_probs.append(whq_order_probs)
        all_whq_order_probs_not_marg.append(whq_order_probs_not_marg)
        new_probs_with_prior = la.probs_of_word_orders(False)
        if ((i+1)%chunk_size == 0 and i+1!=len(train_data) and not ARGS.is_test) or (ARGS.is_test and i+1==len(train_data)):
            if len(la.lf_vocab)==0 or len(la.mwe_vocab)==0 or len(la.shell_lf_vocab)==0 or len(la.sync_vocab)==0:
                w2lf, w2s, w2b = 0, 0, 0
            else:
                _, w2lf, w2s, w2b = test_word_acc()
            if ARGS.test_nonce:
                nonce_results = test_nonce()
                for k,v in nonce_results.items():
                    all_nonce_results[k].append(v)
            all_w2lf_accs.append(w2lf); all_w2syn_accs.append(w2s); all_w2both_accs.append(w2b)
            new_whw_acc, new_whwwgt_acc, preds, preds_wgt, whw_gts, whw_gt_words = test_wh_words(test_data)
            all_whw_preds.append(preds)
            all_whwwgt_preds.append(preds_wgt)
            all_whw_accs.append(new_whw_acc)
            all_whwwgt_accs.append(new_whwwgt_acc)
            if ARGS.test_incr:
                new_acc, new_harsh_acc, new_select_acc, acc_by_type = test_parses(final_chunk)
                all_accs.append(new_acc)
                all_harsh_accs.append(new_harsh_acc)
                all_select_accs.append(new_select_acc)
                for k,v in acc_by_type.items():
                    all_accs_by_type[k].append(v)
                check_dir(f'experiments/{ARGS.expname}/gens')
                if ARGS.test_next_chunk:
                    next_data = data_to_use[i:i+len(final_chunk)]
                    new_next_acc, new_next_harsh_acc, new_next_select_acc, acc_by_type = test_parses(next_data)
                    all_next_accs.append(new_next_acc)
                    all_next_harsh_accs.append(new_next_harsh_acc)
                    all_next_select_accs.append(new_next_select_acc)
        all_word_order_probs.append(new_probs_with_prior)
        la.train()

    time_per_dpoint = (time()-train_start_time)/len(d['data'])
    print(f'Time per dpoint: {time_per_dpoint:.6f}')
    if len(train_data) > 0:
        la.compute_inverse_probs()
    inverse_probs_start_time = time()
    print(f'Time to compute inverse probs: {time()-inverse_probs_start_time:.3f}s')

    if ARGS.jreload_from is not None:
        df_prior = pd.read_csv(f'experiments/{ARGS.jreload_from}/{ARGS.jreload_from}_word_order_probs.csv', index_col=0)
    else:
        la.save_to(f'experiments/{ARGS.expname}')
        print(df_prior:=pd.DataFrame(all_word_order_probs))
        df_prior.to_csv(f'experiments/{ARGS.expname}/{ARGS.expname}_word_order_probs.csv')

    print(whq_order_df:=pd.DataFrame(all_whq_order_probs))
    print(whq_order_df_not_marg:=pd.DataFrame(all_whq_order_probs_not_marg))
    whq_order_df.to_csv(f'experiments/{ARGS.expname}/{ARGS.expname}_whq_order_probs.csv')
    with open(f'experiments/{ARGS.expname}/test-gens.txt', 'w') as f:
        for _ in range(300):
            f.write(la.generate_words() + '\n')
    cs = ['r','g','b','y','orange','brown', 'm', 'c', 'k']
    if ARGS.jreload_from is None:
        results_df, final_word2lf_acc, final_word2syn_acc, final_word2both_acc = test_word_acc()
        all_w2lf_accs.append(final_word2lf_acc); all_w2syn_accs.append(final_word2syn_acc); all_w2both_accs.append(final_word2both_acc)
        final_whw_acc, final_whwwgt_acc, preds, preds_wgt, whw_gts, whw_gt_words = test_wh_words(test_data)
        all_whw_preds.append(preds)
        all_whwwgt_preds.append(preds_wgt)
        all_whw_accs.append(final_whw_acc); all_whwwgt_accs.append(final_whwwgt_acc)
        all_whw_preds.append(whw_gts)
        all_whwwgt_preds.append(whw_gts)
        if not ARGS.no_test_roots:
            final_acc_enws, final_harsh_acc, final_select_acc, acc_by_type = test_parses(final_chunk)
            for k,v in acc_by_type.items():
                all_accs_by_type[k].append(v)
        dist_substr = '' if ARGS.n_distractors==0 else f' dist{ARGS.n_distractors}'
        if ARGS.is_test:
            xplot = np.arange(len(all_w2lf_accs))
        else:
            xplot = [i for i in range(len(train_data[:ARGS.break_train_at])) if (i+1)%chunk_size == 0 and i+1!=len(train_data[:ARGS.break_train_at])]
            xplot.append(len(train_data[:ARGS.break_train_at]))
        whw_preds_df = pd.DataFrame(all_whw_preds, index=xplot+['gt'])
        whw_preds_df_wgt = pd.DataFrame(all_whwwgt_preds, index=xplot+['gt'])
        whw_preds_df.to_csv(f'experiments/{ARGS.expname}/whw-preds.csv')
        whw_preds_df_wgt.to_csv(f'experiments/{ARGS.expname}/whw-preds-wgt.csv')
        whw_preds_df_wgt = pd.DataFrame(all_whwwgt_preds, index=xplot+['gt'])
        #no_single_word_uts_accs = (whw_preds_df == whw_preds_df.loc['gt']).sum(axis=1) / (whw_preds_df.shape[1] - (whw_preds_df == 'Swhq').sum(axis=1))
        #no_single_word_uts_accs_wgt = (whw_preds_df_wgt == whw_preds_df_wgt.loc['gt']).sum(axis=1) / (whw_preds_df_wgt.shape[1] - (whw_preds_df_wgt == 'Swhq').sum(axis=1))
        #plt.plot(xplot, no_single_word_uts_accs, label='without gt', color='g')
        #plt.plot(xplot, no_single_word_uts_accs_wgt, label='with gt', color='r')
        #plt.xlabel('Num Training Points')
        #plt.ylabel(f'Relative Probability')
        #plt.legend(loc='upper left')
        #fpath = f'experiments/{ARGS.expname}/{ARGS.expname}_whw_order_accs1.png'
        #plt.savefig(fpath)
        #plt.clf()
        #os.system(f'/usr/bin/xdg-open {fpath}')

        ix=(whw_preds_df.iloc[:-1] != 'Swhq').all(axis=0)
        nswua2 = [(whw_preds_df.iloc[i][ix]==whw_preds_df.loc['gt'][ix]).mean() for i in range(whw_preds_df.shape[0]-1)]
        ix_wgt=(whw_preds_df_wgt.iloc[:-1] != 'Swhq').all(axis=0)
        nswuawgt2 = [(whw_preds_df_wgt.iloc[i][ix_wgt]==whw_preds_df_wgt.loc['gt'][ix_wgt]).mean() for i in range(whw_preds_df_wgt.shape[0]-1)]
        plt.plot(xplot, nswua2, label='without gt', color='g')
        plt.plot(xplot, nswuawgt2, label='with gt', color='r')
        plt.xlabel('Num Training Points')
        plt.ylabel(f'Accuracy')
        plt.legend(loc='upper left')
        fpath = f'experiments/{ARGS.expname}/{ARGS.expname}_whw_order_accs2.png'
        plt.savefig(fpath)
        plt.clf()
        os.system(f'/usr/bin/xdg-open {fpath}')

        breakpoint()
        plt.plot(xplot, all_w2lf_accs, color='darkorange', label='word meaning')
        plt.plot(xplot, all_w2syn_accs, color='r', label='word category')
        plt.plot(xplot, all_w2both_accs, color='g', label='both')
        plt.xlabel('Num Training Points')
        plt.ylabel(f'Relative Probability')
        plt.legend(loc='upper left')
        plt.title(f'Learned lexical entries{dist_substr}')
        fpath = f'experiments/{ARGS.expname}/{ARGS.expname}_test_word_accs.png'
        plt.savefig(fpath)
        plt.clf()
        os.system(f'/usr/bin/xdg-open {fpath}')

        plt.plot(xplot, all_whw_accs, color='g', label='with gt')
        plt.plot(xplot, all_whwwgt_accs, color='b', label='without gt')
        plt.xlabel('Num Training Points')
        plt.ylabel(f'Accuracy')
        plt.legend(loc='upper left')
        plt.title(f'Predicting the Category of Wh-words')
        fpath = f'experiments/{ARGS.expname}/{ARGS.expname}_test_whw_accs.png'
        plt.savefig(fpath)
        plt.clf()
        os.system(f'/usr/bin/xdg-open {fpath}')
        breakpoint()

        if ARGS.test_nonce:
            final_nonce_results = test_nonce()
            for k,v in final_nonce_results.items():
                all_nonce_results[k].append(v)
            for const_type, colour in zip(nonce_dps.keys(), cs):
                plt.plot(xplot, all_nonce_results[const_type], color=colour, label=const_type)
            plt.xlabel('Num Training Points')
            plt.ylabel(f'Correct Parse Probability (CPP)')
            plt.legend(loc='upper left')
            plt.title(f'One-trial Learning Ability{dist_substr}')
            fpath = f'experiments/{ARGS.expname}/{ARGS.expname}_one-trials.png'
            plt.savefig(fpath)
            plt.clf()
            os.system(f'/usr/bin/xdg-open {fpath}')

        if ARGS.test_incr:
            for (k,v), colour in zip(all_accs_by_type.items(), cs):
                plt.plot(xplot, v, color=colour, label=k)
            plt.xlabel('Num Training Points')
            #plt.ylabel(f'Select Acc')
            plt.ylabel(f'Accuracy')
            plt.legend(loc='upper left')
            plt.title(f'Acc on Final 10% by Utterance Type{dist_substr}')
            fpath = f'experiments/{ARGS.expname}/{ARGS.expname}_acc-by-type.png'
            plt.savefig(fpath)
            plt.clf()
            os.system(f'/usr/bin/xdg-open {fpath}')

            all_accs.append(final_acc_enws); all_harsh_accs.append(final_harsh_acc); all_select_accs.append(final_select_acc)
            plt.plot(xplot, all_accs, color='b', label='no unseen words')
            plt.plot(xplot, all_harsh_accs, color='r', label='all utterances')
            plt.plot(xplot, all_select_accs, color='y', label='select acc')
            plt.xlabel('Num Training Points')
            plt.ylabel('Accuracy')
            plt.legend(loc='upper left')
            plt.title(f'Accuracy on Final 10% of Utterances{dist_substr}')
            fpath = f'experiments/{ARGS.expname}/{ARGS.expname}_test_lf_accs.png'
            plt.savefig(fpath)
            plt.clf()
            os.system(f'/usr/bin/xdg-open {fpath}')

            if ARGS.test_next_chunk:
                all_next_accs.append(final_acc_enws); all_next_harsh_accs.append(final_harsh_acc); all_next_select_accs.append(final_select_acc)
                plt.plot(xplot, all_next_accs, color='b', label='no unseen words next chunk')
                plt.plot(xplot, all_next_harsh_accs, color='r', label='all utterances next chunk')
                plt.plot(xplot, all_next_select_accs, color='y', label='select acc next chunk')
                plt.xlabel('Num Training Points')
                plt.ylabel('Accuracy')
                plt.legend(loc='upper left')
                plt.title(f'Accuracy on Next 5% of Utterances{dist_substr}')
                fpath = f'experiments/{ARGS.expname}/{ARGS.expname}_next_lf_accs.png'
                plt.savefig(fpath)
                plt.clf()
                os.system(f'/usr/bin/xdg-open {fpath}')
        #plt.show()

        def plot_df(df, info=''):
            if info != '':
                info = f' {info}'
            for i,c in enumerate(df.columns):
                plt.plot(np.arange(len(df)), df[c], label=c, color=cs[i])
            plt.legend(loc='upper right')
            plt.xlabel('Num Training Points')
            plt.ylabel('Relative Probability')
            plt.title(f'Word Order Probs{info}')
            check_dir('plotted_figures')
            fpath = f'experiments/{ARGS.expname}/word_order_probs{info.replace(" ","_").lower()}.png'
            plt.savefig(fpath)
            plt.clf()
            if ARGS.show_plots:
                os.system(f'/usr/bin/xdg-open {fpath}')

        if ARGS.expname_for_plot_titles is not None:
            dn_info = ARGS.expname_for_plot_titles
        else:
            dn = ARGS.dset[0].upper() + ARGS.dset[1:]
            dn_info = dn if ARGS.n_distractors==0 else f'{dn} {ARGS.n_distractors} distractors'
            dn_info = f'{dn_info} {ARGS.expname[:-1]}'.replace('_',' ')
        if ARGS.jreload_from is None:
            plot_df(df_prior, dn_info)
            plot_df(whq_order_df, 'Whq words only')
            plot_df(whq_order_df_not_marg, 'Whq words only not marg')
        results_df.to_csv(f'experiments/{ARGS.expname}/{ARGS.expname}_full_preds_and_scores.csv')
    else:
        whw_preds_df = pd.read_csv(f'experiments/{ARGS.jreload_from}/whw-preds.csv', index_col=0)
        whw_preds_df_wgt = pd.read_csv(f'experiments/{ARGS.jreload_from}/whw-preds-wgt.csv', index_col=0)
        final_whw_acc, final_whwwgt_acc, preds, preds_wgt, whw_gts, whw_gt_words = test_wh_words(test_data)

    whw_preds_df.to_csv(f'experiments/{ARGS.expname}/whw-preds.csv')
    whw_preds_df_wgt.to_csv(f'experiments/{ARGS.expname}/whw-preds-wgt.csv')

    def get_changes(df):
        for i in range(df.shape[0]-1):
            correct_idx = df.iloc[i] == df.loc['gt']
            #corrects = df.iloc[i][correct_idx].values
            if i > 0:
                prev_wrong_now_right_idx = correct_idx & ~prev_correct_idx
                for j, x in enumerate(prev_wrong_now_right_idx):
                    if x and df.iloc[i-1,j] != 'Swhq':
                        print('New corrects:', df.iloc[i-1:i+1,j].values, 'at', df.index[i], whw_gt_words[j])
                prev_right_now_wrong_idx = ~correct_idx & prev_correct_idx
                for j, x in enumerate(prev_right_now_wrong_idx):
                    if x and df.iloc[i,j] != 'Swhq':
                        print('New incorrects:', df.iloc[i-1:i+1,j].values, 'at', df.index[i], whw_gt_words[j])
            n_correct = (correct_idx).sum()
            denom = (df.iloc[i] != 'Swhq').sum()
            n_correct / denom
            prev_correct_idx = correct_idx

    print('WITHOUT GT')
    get_changes(whw_preds_df)
    print('WITH GT')
    get_changes(whw_preds_df_wgt)
    breakpoint()

    la.parse("what wo n't you eat".split())
    test_wh_words(test_data)
    if ARGS.print_word_acc:
        print(results_df)
    if ARGS.db_after:
        breakpoint()
    if ARGS.jreload_from is None:
        with open(f'experiments/{ARGS.expname}/{ARGS.expname}_summary.txt','w') as f:
            file_print(f'Total run time: {time()-start_time:.3f}s',f)
            file_print(f'Word to LF accuracy: {final_word2lf_acc:.4f}',f)
            file_print(f'Word to syncat accuracy: {final_word2syn_acc:.4f}',f)
            file_print(f'Full lexical accuracy: {final_word2both_acc:.4f}',f)
            if len(train_data) == 0:
                file_print('Trainset was empty, no word_order_probs logged',f)
            else:
                file_print('Final word order probs:',f)
                for k,v in all_word_order_probs[-1].items():
                    file_print(f'{k}: {v:.6f}',f)
