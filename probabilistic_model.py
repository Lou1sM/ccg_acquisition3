import numpy as np
from gt_parse_graphs import gts
import os
import networkx as nx
import matplotlib.pyplot as plt
import math
from pprint import pprint
from copy import copy
from dl_utils.misc import set_experiment_dir
from os.path import join
import pandas as pd
from utils import file_print, get_combination, is_direct_congruent, is_fit_by_type_raise, combine_lfs, logical_type_raise, maybe_de_type_raise, split_respecting_brackets, n_lambda_binders, possible_syn_cats, infer_slash
from time import time
import argparse
from abc import ABC
import re
from parser import LogicalForm, ParseNode
import json


def type_raise(cat,direction,out_cat='S'):
    if direction == 'fwd':
        return f'{out_cat}/({out_cat}\\{cat})'
    elif direction == 'bck':
        return f'{out_cat}\\({out_cat}/{cat})'
    elif direction == 'sem':
        return f'{out_cat}|({out_cat}|{cat})'

class DirichletProcess():
    def __init__(self,alpha):
        self.memory = {'count':0}
        self.alpha = alpha

    def observe(self,obs):
        if obs in self.memory:
            self.memory[obs] += 1
        else:
            self.memory = self.memory | {obs:1}
        self.memory['count'] += 1
        assert np.allclose(2*self.memory['count'],sum(self.memory.values()),rtol=1e-6)

    def prob(self,obs):
        base_prob = 0.25 * (1/12)**(obs.count('/') + obs.count('\\') + obs.count('|'))
        mx = self.memory.get(obs,0)
        return (mx+self.alpha*base_prob)/(self.memory['count']+self.alpha)

class BaseDirichletProcessLearner(ABC):
    def __init__(self,alpha):
        self.alpha = alpha
        self.memory = {}
        self.base_distribution_cache = {}
        self.buffer = []

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
        return (mx+self.alpha*base_prob)/(self.memory[x]['count']+self.alpha)

    def prob(self,*args,**kwargs):
        return self._prob(*args,**kwargs)

    def _observe(self,y,x,weight):
        if x not in self.memory:
            self.memory[x] = {y:weight,'count':weight}
        elif y not in self.memory[x]:
            self.memory[x] = self.memory[x] | {y:weight}
            self.memory[x]['count'] += weight
        else:
            self.memory[x][y] += weight
            self.memory[x]['count'] += weight
        assert np.allclose(2*self.memory[x]['count'],sum(self.memory[x].values()),rtol=1e-6)

    def observe(self,*args,**kwargs):
        """Can be overwritten if necessary."""
        return self._observe(*args,**kwargs)

    def flush_buffer(self):
        for _ in range(len(self.buffer)):
            self.observe(*self.buffer.pop())
        assert len(self.buffer) == 0

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
        options,unnormed_probs = zip(*[z for z in self.memory[x].items() if z[0]!='count'])
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

class CCGDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        assert x != 'NP + S/NP\\NP'
        #n_slashes = len(re.findall(r'[\\/\|]',x))
        #return (0.5555)*0.9**(n_slashes+1) # Omri 2017 had 0.2
        return 1

    def observe(self,y,x,weight):
        if y == 'leaf':
            self._observe(y,maybe_de_type_raise(x),weight)
        else:
            self._observe(y,x,weight)

    def prob(self,y,x):
        if y != 'leaf':
            outcat, _ = get_combination(*y.split(' + '))
            if outcat is None: #only time this should happen is during inference with crossed cmp
                return 0
            if not is_direct_congruent(outcat,x):
                return 0
        base_prob = self.base_distribution(y)
        if x not in self.memory:
            if y == 'leaf':
                #return base_prob/(self.memory.get('count',0) + 1) #could be +base_prob instead of 1
                return base_prob
            return base_prob
        mx = self.memory[x].get(y,0)
        return (mx+self.alpha*base_prob)/(self.memory[x]['count']+self.alpha)

class ShellMeaningDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        n_vars = len(set(re.findall(r'(?<!lambda )\$\d',x)))
        n_constants = float('const' in x)+float('quant' in x)
        norm_factor = (np.e+1)/(np.e**2 - np.e - 1)
        return norm_factor * np.e**(-2*n_vars - n_constants)

    def prob(self,y,x):
        if len(split_respecting_brackets(x,sep=['/','\\','|'])) != n_lambda_binders(y)+1:
            print(y,x)
            return 0
        #y = maybe_de_type_raise(y_)
        base_prob = self.base_distribution(y)
        if x not in self.memory:
            return base_prob
        mx = self.memory[x].get(y,0)
        return (mx+self.alpha*base_prob)/(self.memory[x]['count']+self.alpha)

class MeaningDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        n_vars = len(set(re.findall(r'(?<!lambda )\$\d',x)))
        n_constants = len(set([z for z in re.findall(r'[a-z]*',x) if 'lambda' not in z and len(z)>0]))
        norm_factor = (np.e+1)/(np.e**2 - np.e - 1)
        return norm_factor * np.e**(-2*n_vars - n_constants)

class WordSpanDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution(self,x):
        assert len(x) > 0
        return 28**-math.ceil(len(x)/2) # kinda approximate num phonetic chunks

class LanguageAcquirer():
    def __init__(self, base_lexicon):
        self.base_lexicon = base_lexicon
        self.syntaxl = CCGDirichletProcessLearner(10)
        self.shmeaningl = ShellMeaningDirichletProcessLearner(1)
        self.meaningl = MeaningDirichletProcessLearner(1)
        self.wordl = WordSpanDirichletProcessLearner(0.25)
        self.full_lfs_cache = {} # maps lf_strings to LogicalForm objects
        self.lf_parts_cache = {'splits':{},'cats':{}} # maps LogicalForm objects to lists of (left-child,right-child)
        self.parse_node_cache = {} # maps utterances (str) to ParseNode objects, including splits
        self.root_sem_cat_memory = DirichletProcess(1) # counts of the sem_cats it's seen as roots

    @property
    def lf_vocab(self):
        return self.wordl.memory.keys()

    @property
    def shell_lf_vocab(self):
        return self.meaningl.memory.keys()

    @property
    def sem_cat_vocab(self):
        return self.shmeaningl.memory.keys()

    @property
    def syn_cat_vocab(self):
        return self.syntaxl.memory.keys()

    @property
    def splits_vocab(self):
        return list(set([w for x in self.syntaxl.memory.values() for w in x.keys() if w!='count']))

    @property
    def mwe_vocab(self):
        return list(set([w for x in self.wordl.memory.values() for w in x.keys() if w!='count']))

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

        self.word_to_lf_probs = pd.read_pickle(join(fpath,'word_to_lf_probs.pkl'))
        self.lf_to_lf_shell_probs = pd.read_pickle(join(fpath,'lf_to_lf_shell_probs.pkl'))
        self.lf_shell_to_sem_probs = pd.read_pickle(join(fpath,'lf_shell_to_sem_probs.pkl'))
        self.word_to_sem_probs = pd.read_pickle(join(fpath,'word_to_sem_probs.pkl'))

    def save_to(self,fpath):
        to_dump = {'syntax': {'memory':self.syntaxl.memory,
                   'base_distribution_cache':self.syntaxl.base_distribution_cache},
                  'shmeaningl': {'memory':self.shmeaningl.memory,
                   'base_distribution_cache':self.shmeaningl.base_distribution_cache},
                  'meaning': {'memory':self.meaningl.memory,
                   'base_distribution_cache':self.meaningl.base_distribution_cache},
                  'word': {'memory':self.wordl.memory,
                   'base_distribution_cache':self.wordl.base_distribution_cache}}

        distributions_fpath = join(fpath,'distributions.json')
        with open(distributions_fpath,'w') as f:
            json.dump(to_dump,f)

        self.word_to_lf_probs.to_pickle(join(fpath,'word_to_lf_probs.pkl'))
        self.lf_to_lf_shell_probs.to_pickle(join(fpath,'sem_to_sem_shell.pkl'))
        self.lf_shell_to_sem_probs.to_pickle(join(fpath,'lf_shell_to_sem_probs.pkl'))

    def flush_buffers(self):
        self.syntaxl.flush_buffer()
        self.shmeaningl.flush_buffer()
        self.meaningl.flush_buffer()
        self.wordl.flush_buffer()

    def show_word_meanings(self,word): # prob of meaning given word assuming flat prior over meanings
        distr = self.wordl.inverse_distribution(word)
        probs = sorted(distr.items(), key=lambda x:x[1])[-15:]
        print(f'\nLearned Meaning for \'{word}\'')
        print('\n'.join([f'{prob:.3f}: {word}' for word,prob in probs]))

    def show_word(self,word,f=None):
        meanings = self.word_to_lf_probs.loc[word].sort_values()[-10:]
        file_print(f'\nLearned meanings for \'{word}\'',f)
        file_print('\n'.join([f'{word}: {100*prob:.2f}%' for word,prob in meanings.items() if prob > 1e-4]),f)

        syn_cats = self.word_to_sem_probs.loc[word].sort_values()[-10:]
        file_print(f'\nLearned syntactic categories for \'{word}\'',f)
        file_print('\n'.join([f'{word}: {100*prob:.2f}%' for word,prob in syn_cats.items() if prob > 1e-4]),f)

    def show_splits(self,syn_cat,f):
        file_print(f'Learned splits for category {syn_cat}',f)
        counts = self.syntaxl.memory[syn_cat]
        norm = counts['count']
        probs = {k:counts[k]/norm for k in sorted(counts,key=lambda x:counts[x]) if k!='count'}
        file_print('\n'.join([f'{prob:.3f}: {word}' for word,prob in probs.items()]),f)

    def get_lf(self,lf_str):
        if lf_str in self.full_lfs_cache:
            return self.full_lfs_cache[lf_str]
        else:
            lf = LogicalForm(lf_str,self.base_lexicon,caches=self.lf_parts_cache,parent='START')
            self.full_lfs_cache[lf_str] = lf
            return lf

    def train_one_step(self,lf_str,words):
        root = self.make_parse_node(lf_str,words) # words is a list
        prob_cache = {}
        root_prob = root.propagate_below_probs(self.syntaxl,self.shmeaningl,
                       self.meaningl,self.wordl,prob_cache,split_prob=1,is_map=False)
        if root_prob==0:
            breakpoint()
        root.propagate_above_probs(1)
        #if len(words) == 4 and words[0] == 'does':
        #if ' '.join(words) == 'the mountain deracinates seattle':
        #if 'deracinate' in ' '.join(words):
            #breakpoint()
            #root.logical_form.subtree_string_(as_shell=True,recompute=True,show_treelike=False)
        for node, prob in prob_cache.items():
            if node.parent is not None and not node.is_g:
                if node.is_fwd:
                    syntax_split = node.syn_cat + ' + ' + node.sibling.syn_cat
                else:
                    syntax_split = node.sibling.syn_cat + ' + ' + node.syn_cat
                update_weight = node.below_prob * node.above_prob / root_prob # for conditional
                self.syntaxl.observe(syntax_split,node.parent.syn_cat,weight=update_weight)
            leaf_prob = node.above_prob*node.stored_prob_as_leaf/root_prob
            if not leaf_prob > 0:
                print(node)
            lf = node.logical_form.subtree_string(alpha_normalized=True,recompute=True)
            word_str, lf, shell_lf, sem_cat, syn_cat = node.info_if_leaf()
            self.syntaxl.buffer.append(('leaf',syn_cat,leaf_prob))
            #if shell_lf == 'lambda $0.quant $0' and sem_cat == 'NP|N':
                #breakpoint()
            self.shmeaningl.buffer.append((shell_lf,sem_cat,leaf_prob))
            self.meaningl.buffer.append((lf,shell_lf,leaf_prob))
            self.wordl.buffer.append((word_str,lf,leaf_prob))
        #print(words,sum([x[2] for x in self.wordl.buffer]))
        self.flush_buffers()

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

    def make_parse_node(self,lf_str,words):
        lf = self.get_lf(lf_str)
        if ' '.join([lf_str]+words) in self.parse_node_cache:
            parse_root = self.parse_node_cache[' '.join([lf_str]+words)]
        else:
            parse_root = ParseNode(lf,words,'ROOT')
            self.root_sem_cat_memory.observe(parse_root.sem_cat)
            self.parse_node_cache[' '.join([lf_str]+words)] = parse_root
        return parse_root

    def compute_inverse_probs(self): # prob of meaning given word assuming flat prior over meanings
        self.word_to_lf_probs = pd.DataFrame([{y:self.wordl.prob(x,y) for y in self.lf_vocab} for x in self.mwe_vocab],index=self.mwe_vocab)

        self.lf_to_lf_shell_probs = pd.DataFrame([{y:self.meaningl.prob(x,y) for y in self.shell_lf_vocab} for x in self.lf_vocab],index=self.lf_vocab)
        self.lf_shell_to_sem_probs = pd.DataFrame([{y:self.shmeaningl.prob(x,y) for y in self.sem_cat_vocab} for x in self.shell_lf_vocab],index=self.shell_lf_vocab)

    def leaf_probs_of_word_span(self,words,beam_size):
        if words not in self.mwe_vocab and ' ' not in words:
            print(f'\'{words}\' not seen before as a leaf')
        beam = [{'lf':lf,'prob': self.wordl.prob(words,lf)} for lf in self.lf_vocab]
        beam = sorted(beam,key=lambda x:x['prob'])[-beam_size:]
        beam = [dict(b,shell_lf=shell_lf,prob=b['prob']*self.meaningl.prob(b['lf'],shell_lf))
            for shell_lf in self.shell_lf_vocab for b in beam]
        beam = sorted(beam,key=lambda x:x['prob'])[-beam_size:]
        beam = [dict(b,sem_cat=sem_cat,prob=b['prob']*self.shmeaningl.prob(b['shell_lf'],sem_cat))
            for sem_cat in self.sem_cat_vocab for b in beam]
        beam = sorted(beam,key=lambda x:x['prob'])[-beam_size:]
        beam = [dict(b,prob=b['prob']*self.syntaxl.prob('leaf',b['sem_cat'])) for b in beam]
        beam = sorted(beam,key=lambda x:x['prob'])[-beam_size:]
        return [dict(b,words=words,rule='leaf',backpointer=None) for b in beam]

    def parse(self,words):
        N = len(words)
        beam_size = 50
        probs_table = np.empty((N,N),dtype='object') #(i,j) will be a dict of len(beam_size) saying probs of top syn_cats
        for i in range(N):
            probs_table[0,i] = self.leaf_probs_of_word_span(words[i],beam_size)

        def add_prob_of_span(i,j):
            possible_nexts = self.leaf_probs_of_word_span(' '.join(words[j:j+i]),beam_size)
            for k in range(1,i):
                left_chunk_probs = probs_table[k-1,j]
                right_chunk_probs = probs_table[i-k-1,j+k] # total len is always i, -1s bc 0-index
                for left_idx, left_option in enumerate(left_chunk_probs):
                    for right_idx, right_option in enumerate(right_chunk_probs):
                        l_cat, r_cat = left_option['sem_cat'], right_option['sem_cat']
                        #should_type_raise, outcat = is_fit_by_type_raise(lsyn_cat,rsyn_cat)
                        #if should_type_raise:
                            #lsyn_cat = type_raise(lsyn_cat,'fwd',outcat)
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
                            prob = left_option['prob']*right_option['prob']*self.syntaxl.prob(split,csync)
                            backpointer = (k,j,left_idx), (i-k,j+k,right_idx)
                            #backpointer contains the coordinates in probs_table, and the idx in the
                            #beam, of the two locations that the current one could be split into
                            pn = {'sem_cat':combined,'lf':lf,'backpointer':backpointer,'rule':rule,'prob':prob,'words':' '.join(words[j:j+i]),'syn_cat':csync}
                            possible_nexts.append(pn)
            if i == N:
                for pn in possible_nexts:
                    pn['syn_cat'] = pn['sem_cat']
                    pn['prob'] = pn['prob']*self.root_sem_cat_memory.prob(pn['syn_cat'])
            to_add =sorted(possible_nexts,key=lambda x:x['prob'])[-beam_size:]
            probs_table[i-1,j] = to_add

        #if words == 'maryland buys a dog'.split():
            #breakpoint()
        for a in range(2,N+1):
            for b in range(N-a+1):
                add_prob_of_span(a,b)
        if len(probs_table[N-1,0]) == 0:
            return 'No parse found'

        def show_parse(num):
            syntax_tree_level = [dict(probs_table[N-1,0][num],idx='1',hor_pos=0)]
            #syntax_tree_level[0]['hor_pos'] = len(syntax_tree_level[0]['words'].split())/2
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
                    left_split = dict(left_split,idx=item['idx'][:-1] + '01',syn_cat=lsync)
                                        #hor_pos=left_pos+(left_len-1)/2 - (N-1)/2)
                    right_split = dict(right_split,idx=item['idx'][:-1] + '21', syn_cat=rsync)
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
        self.draw_graph(favourite_all_syntax_tree_levels)

        if ARGS.db_parse:
            #self.make_parse_node('Q (talk (a lake))','does a lake talk'.split()).possible_split
            breakpoint()

    def draw_graph(self,all_syntax_tree_levels,is_gt=False):
        leaves = [n for level in all_syntax_tree_levels for n in level if n['rule']=='leaf']
        leaves.sort(key=lambda x:x['idx'])
        G=nx.Graph()
        all_words = all_syntax_tree_levels[0][0]['words'].split()
        for i in range(len(all_words)):
            if all_words.count(all_words[i]) > 1:
                matching_idxs = [idx for idx,w in enumerate(words) if w == all_words[i]]
                for suffix,midx in matching_idxs:
                    all_words[midx] = all_words[midx] + suffix

        def _wp(w):
            return all_words.index(w) - (len(all_words)-1)/2

        def _wps(words):
            return sum([_wp(w) for w in words.split()])/len(words.split())

        for level in all_syntax_tree_levels:
            for x in level:
                x['hor_pos'] = _wps(x['words'])
        for j,syntax_tree_level in enumerate(all_syntax_tree_levels):
            wrong_avg = sum([x['hor_pos'] for x in syntax_tree_level])/len(syntax_tree_level)
            words_at_level = ' '.join([x['words'] for x in syntax_tree_level])
            what_avg_should_be = _wps(words_at_level)
            #print(f'"{words_at_level}" out of {all_words} wrong: {wrong_avg} right: {what_avg_should_be}')
            correction = what_avg_should_be - wrong_avg
            for i,node in enumerate(syntax_tree_level):
                hor_pos = node['hor_pos']
                if node['rule'] != 'leaf' and j != 0: # keep leaves and root exactly on their bin
                    hor_pos += correction/2
                    assert hor_pos <= max([x['hor_pos'] for x in leaves])
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
        plt.savefig(f'plotted_graphs/{fname}.png')
        plt.clf()
        if ARGS.show_graphs:
            os.system(f'/usr/bin/xdg-open plotted_graphs/{fname}.png')

if __name__ == "__main__":
    ARGS = argparse.ArgumentParser()
    ARGS.add_argument("--expname", type=str,default='tmp')
    ARGS.add_argument("--reload_from", type=str)
    ARGS.add_argument("--n_test", type=int,default=5)
    ARGS.add_argument("--n_dpoints", type=int, default=-1)
    ARGS.add_argument("--db_at", type=int, default=-1)
    ARGS.add_argument("--max_sent_len", type=int, default=9)
    ARGS.add_argument("--n_epochs", type=int, default=1)
    ARGS.add_argument("--n_generate", type=int, default=0)
    ARGS.add_argument("-tt","--short_test_run", action="store_true")
    ARGS.add_argument("-t","--test_run", action="store_true")
    ARGS.add_argument("--show_splits", action="store_true")
    ARGS.add_argument("--show_graphs", action="store_true")
    ARGS.add_argument("--db_parse", action="store_true")
    ARGS.add_argument("--db_after", action="store_true")
    ARGS.add_argument("--overwrite", action="store_true")
    ARGS.add_argument("--test_gts", action="store_true")
    ARGS.add_argument("--shuffle", action="store_true")
    ARGS.add_argument("-d", "--dset", type=str, default='simplified_adam')
    ARGS.add_argument("--cat_to_sample_from", type=str, default='S')
    ARGS = ARGS.parse_args()

    ARGS.test_run = ARGS.test_run or ARGS.short_test_run

    if ARGS.test_run:
        ARGS.expname = 'tmp'
        if ARGS.n_dpoints == -1:
            ARGS.n_dpoints = 10
    elif ARGS.expname is None:
        expname = f'{ARGS.n_epochs}_{ARGS.max_sent_len}'
    else:
        expname = ARGS.expname
    if ARGS.short_test_run:
        ARGS.n_dpoints = 30
    set_experiment_dir(f'experiments/{ARGS.expname}',overwrite=ARGS.overwrite,name_of_trials='experiments/tmp')
    with open(f'data/{ARGS.dset}.json') as f: d=json.load(f)

    #NAMES = d['np_list'] + [str(x) for x in range(1,11)] # because of the numbers in simple dsets
    #NOUNS = d['nouns'] + [x+'-s' for x in d['nouns']] # for pl., quicker than stripping each lookup
    #TRANSITIVES = d['transitive_verbs']
    #INTRANSITIVES = d['intransitive_verbs']
    #base_lexicon = {w:cat for item,cat in zip([NAMES,NOUNS,INTRANSITIVES,TRANSITIVES],('NP','N','S|NP','S|NP|NP')) for w in item}
    base_lexicon = {'you':'NP','me':'NP','he':'NP','she':'NP','it':'NP'}

    language_acquirer = LanguageAcquirer(base_lexicon)
    if ARGS.reload_from is not None:
        language_acquirer.load_from(f'experiments/{ARGS.reload_from}')
    if ARGS.shuffle: np.random.shuffle(d['data'])
    NDPS = len(d['data']) if ARGS.n_dpoints == -1 else ARGS.n_dpoints
    f = open(f'experiments/{ARGS.expname}/results.txt','w')
    all_data = d['data'][:NDPS]
    train_data = all_data[:-len(all_data)//5]
    test_data = train_data if ARGS.test_run else all_data[-len(all_data)//5:]
    if 'questions' in ARGS.dset:
        univ = {'words': ['does', 'a', 'lake', 'talk'], 'lf': 'Q (talk (a lake))'}
        train_data = train_data[:10] + [univ] + train_data[10:]
        test_data = [univ] + test_data
    start_time = time()
    for epoch_num in range(ARGS.n_epochs):
        epoch_start_time = time()
        for i,dpoint in enumerate(train_data):
            if i < 10 and epoch_num > 0:
                continue
            words, lf_str = dpoint['words'], dpoint['lf']
            if words[-1] == '?':
                words = words[:-1]
            if i == ARGS.db_at:
                breakpoint()
            if len(words) <= ARGS.max_sent_len:
                language_acquirer.train_one_step(lf_str,words)
        time_per_dpoint = (time()-epoch_start_time)/len(d['data'])
        print(f'Time per dpoint: {time_per_dpoint:.6f}')
        print(f"Epoch {epoch_num} time: {time()-epoch_start_time:.3f} per dpoint: {time_per_dpoint:.6f}")
        language_acquirer.show_splits(ARGS.cat_to_sample_from,f)
        final_parses = {}
        n_correct_parses = 0

    inverse_probs_start_time = time()
    language_acquirer.compute_inverse_probs()
    print(f'Time to compute inverse probs: {time()-inverse_probs_start_time:.3f}s')
    #file_print(f'Accuracy at meaning of state names: {meaning_acc:.1f}%',f)
    #file_print(f'Accuracy at syn-cat of state names: {syn_acc:.1f}%',f)
    if ARGS.n_generate > 0:
        file_print(f'\nSamples for type {ARGS.cat_to_sample_from}:',f)
    for _ in range(ARGS.n_generate):
        generated = language_acquirer.generate_words(ARGS.cat_to_sample_from)
        if any([x['words'][:-1]==generated.split() for x in d['data']]):
            file_print(f'{generated}: seen during training',f)
        else:
            file_print(f'{generated}: not seen during training',f)
    file_print(f'Total run time: {time()-start_time:.3f}s',f)
    f.close()
    for dpoint in test_data[:ARGS.n_test]:
        language_acquirer.parse(dpoint['words'])
    if ARGS.test_gts:
        for sent,gt in gts.items():
            language_acquirer.parse(gt[0][0]['words'].split())
            language_acquirer.draw_graph(gt,is_gt=True)
    print(language_acquirer.syntaxl.memory['S/NP'])
    if ARGS.db_after:
        la = language_acquirer
        breakpoint()
    language_acquirer.save_to(f'experiments/{ARGS.expname}')
