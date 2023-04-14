import numpy as np
from pprint import pprint
from copy import copy
from dl_utils.misc import set_experiment_dir
from os.path import join
import pandas as pd
from utils import file_print, get_combination, is_direct_congruent, is_fit_by_type_raise, combine_lfs, logical_type_raise, maybe_de_type_raise
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
        base_prob = 0.5 # hard-coding for now as one of {S,S_q}
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

    def prob(self,y,x):
        base_prob = self.base_distribution(y)
        if x not in self.memory:
            return base_prob
        mx = self.memory[x].get(y,0)
        return (mx+self.alpha*base_prob)/(self.memory[x]['count']+self.alpha)

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
        if x == 'NP + S/NP\\NP':
            print('observing weird SVO')
            return 0
        num_slashes = len(re.findall(r'[\\/\|]',x))
        return (1.1)*0.9**(num_slashes+1) # Omri 2017 had 0.2

    def observe(self,y,x,weight):
        if y == 'leaf':
            self._observe(y,maybe_de_type_raise(x),weight)
        else:
            self._observe(y,x,weight)

class ShellMeaningDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        num_vars = len(set(re.findall(r'(?<!lambda )\$\d',x)))
        num_constants = float('PLACE' in x)+float('QUANT' in x)
        norm_factor = (np.e+1)/(np.e**2 - np.e - 1)
        return norm_factor * np.e**(-2*num_vars - num_constants)

class MeaningDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        num_vars = len(set(re.findall(r'(?<!lambda )\$\d',x)))
        num_constants = len(set([z for z in re.findall(r'[a-z]*',x) if 'lambda' not in z and len(z)>0]))
        norm_factor = (np.e+1)/(np.e**2 - np.e - 1)
        return norm_factor * np.e**(-2*num_vars - num_constants)

class WordSpanDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution(self,x):
        assert len(x) > 0
        return 28**(-len(x)+1)

class LanguageAcquirer():
    def __init__(self,base_lexicon):
        self.base_lexicon = base_lexicon
        self.syntax_learner = CCGDirichletProcessLearner(100)
        self.shell_meaning_learner = ShellMeaningDirichletProcessLearner(1)
        self.meaning_learner = MeaningDirichletProcessLearner(1)
        self.word_learner = WordSpanDirichletProcessLearner(1)
        self.full_lfs_cache = {} # maps lf_strings to LogicalForm objects
        self.lf_parts_cache = {'splits':{},'cats':{}} # maps LogicalForm objects to lists of (left-child,right-child)
        self.parse_node_cache = {} # maps utterances (str) to ParseNode objects, including splits
        self.root_sem_cat_memory = DirichletProcess(1) # counts of the sem_cats it's seen as roots

    @property
    def lf_vocab(self):
        return self.word_learner.memory.keys()

    @property
    def shell_lf_vocab(self):
        return self.meaning_learner.memory.keys()

    @property
    def sem_cat_vocab(self):
        return self.shell_meaning_learner.memory.keys()

    @property
    def syn_cat_vocab(self):
        return self.syntax_learner.memory.keys()

    @property
    def splits_vocab(self):
        return list(set([w for x in self.syntax_learner.memory.values() for w in x.keys() if w!='count']))

    @property
    def mwe_vocab(self):
        return list(set([w for x in self.word_learner.memory.values() for w in x.keys() if w!='count']))

    @property
    def vocab(self):
        return list(set([w for x in self.word_learner.memory.values()
                        for w in x.keys() if ' ' not in w]))

    def load_from(self,fpath):
        distributions_fpath = join(fpath,'distributions.json')
        with open(distributions_fpath) as f:
            to_load = json.load(f)
        self.syntax_learner.set_from_dict(to_load['syntax'])
        self.shell_meaning_learner.set_from_dict(to_load['shell_meaning'])
        self.meaning_learner.set_from_dict(to_load['meaning'])
        self.word_learner.set_from_dict(to_load['word'])

        self.word_to_lf_probs = pd.read_pickle(join(fpath,'word_to_lf_probs.pkl'))
        self.lf_to_lf_shell_probs = pd.read_pickle(join(fpath,'lf_to_lf_shell_probs.pkl'))
        self.lf_shell_to_sem_probs = pd.read_pickle(join(fpath,'lf_shell_to_sem_probs.pkl'))
        self.word_to_sem_probs = pd.read_pickle(join(fpath,'word_to_sem_probs.pkl'))

    def save_to(self,fpath):
        to_dump = {'syntax': {'memory':self.syntax_learner.memory,
                   'base_distribution_cache':self.syntax_learner.base_distribution_cache},
                  'shell_meaning': {'memory':self.shell_meaning_learner.memory,
                   'base_distribution_cache':self.shell_meaning_learner.base_distribution_cache},
                  'meaning': {'memory':self.meaning_learner.memory,
                   'base_distribution_cache':self.meaning_learner.base_distribution_cache},
                  'word': {'memory':self.word_learner.memory,
                   'base_distribution_cache':self.word_learner.base_distribution_cache}}

        distributions_fpath = join(fpath,'distributions.json')
        with open(distributions_fpath,'w') as f:
            json.dump(to_dump,f)

        self.word_to_lf_probs.to_pickle(join(fpath,'word_to_lf_probs.pkl'))
        self.lf_to_lf_shell_probs.to_pickle(join(fpath,'sem_to_sem_shell.pkl'))
        self.lf_shell_to_sem_probs.to_pickle(join(fpath,'lf_shell_to_sem_probs.pkl'))
        self.word_to_sem_probs.to_pickle(join(fpath,'word_to_sem_probs.pkl'))

    def flush_buffers(self):
        self.syntax_learner.flush_buffer()
        self.shell_meaning_learner.flush_buffer()
        self.meaning_learner.flush_buffer()
        self.word_learner.flush_buffer()

    def show_word_meanings(self,word): # prob of meaning given word assuming flat prior over meanings
        distr = self.word_learner.inverse_distribution(word)
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
        counts = self.syntax_learner.memory[syn_cat]
        norm = counts['count']
        probs = {k:counts[k]/norm for k in sorted(counts,key=lambda x:counts[x]) if k!='count'}
        file_print('\n'.join([f'{prob:.3f}: {word}' for word,prob in probs.items()]),f)

    def get_lf(self,lf_str):
        if lf_str in self.full_lfs_cache:
            return self.full_lfs_cache[lf_str]
        else:
            lf = LogicalForm(lf_str,base_lexicon=self.base_lexicon,caches=self.lf_parts_cache,parent='START')
            self.full_lfs_cache[lf_str] = lf
            return lf

    def train_one_step(self,lf_str,words):
        root = self.make_parse_node(lf_str,words)
        prob_cache = {}
        root_prob = root.propagate_below_probs(self.syntax_learner,self.shell_meaning_learner,
                       self.meaning_learner,self.word_learner,prob_cache,split_prob=1,is_map=False)
        root.propagate_above_probs(1)
        #print(words)
        #if len(words) == 4 and words[0] == 'does':
        #    breakpoint()
        for node, prob in prob_cache.items():
            if node.parent is not None and not node.is_g:
                if node.is_fwd:
                    syntax_split = node.syn_cat + ' + ' + node.sibling.syn_cat
                else:
                    syntax_split = node.sibling.syn_cat + ' + ' + node.syn_cat
                update_weight = node.below_prob * node.above_prob / root_prob # for conditional
                self.syntax_learner.observe(syntax_split,node.parent.syn_cat,weight=update_weight)
            leaf_prob = node.above_prob*node.stored_prob_as_leaf/root_prob
            assert leaf_prob > 0
            lf = node.logical_form.subtree_string(alpha_normalized=True,recompute=True)
            word_str, lf, shell_lf, sem_cat, syn_cat = node.info_if_leaf()
            self.syntax_learner.buffer.append(('leaf',syn_cat,leaf_prob))
            self.shell_meaning_learner.buffer.append((shell_lf,sem_cat,leaf_prob))
            self.meaning_learner.buffer.append((lf,shell_lf,leaf_prob))
            self.word_learner.buffer.append((word_str,lf,leaf_prob))
        self.flush_buffers()

    def test_NPs(self):
        meaning_corrects = 0
        syn_corrects = 0
        attempts = 0
        nps = [x for x,c in self.base_lexicon.items() if c=='NP']
        for w in nps:
            w_spaces = w.replace('_',' ')
            if w_spaces not in self.word_to_lf_probs.index:
                if w_spaces not in ['virginia','kansas','montgomery']:
                    if not not any([w_spaces in ' '.join(x['words']) and len([z for z in x['words'] if z != '?']) <= ARGS.max_sent_len for x in d['data'][:NDPS]]):
                        print(f'\'{w_spaces}\' not here')
                continue
            meaning_pred = self.word_to_lf_probs.loc[w_spaces].idxmax()
            syn_pred = self.word_to_sem_probs.loc[w_spaces].idxmax()
            if meaning_pred==w:
                meaning_corrects += 1
            if syn_pred=='NP':
                syn_corrects += 1
            attempts += 1
        meaning_acc = 100*meaning_corrects/attempts
        syn_acc = 100*syn_corrects/attempts
        return meaning_acc, syn_acc

    def generate_words(self,syn_cat):
        while True:
            split = self.syntax_learner.conditional_sample(syn_cat)
            if split == 'leaf':
                break
            if all([s in self.syntax_learner.memory for s in split.split(' + ') ]):
                break
        sem_cat = re.sub(r'[\\/]','|',syn_cat)
        if split=='leaf':
            shell_lf = self.shell_meaning_learner.conditional_sample(sem_cat)
            lf = self.meaning_learner.conditional_sample(shell_lf)
            return self.word_learner.conditional_sample(lf)
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
        base = pd.Series({x:self.meaning_learner.base_distribution(x) for x in self.lf_vocab})
        self.word_to_lf_probs = self.word_learner.all_inverse_distributions(self.mwe_vocab,base)

        base = pd.Series({x:self.shell_meaning_learner.base_distribution(x) for x in self.shell_lf_vocab})
        self.lf_to_lf_shell_probs = self.meaning_learner.all_inverse_distributions(self.lf_vocab,base)
        base = pd.Series({x:self.syntax_learner.base_distribution(x) for x in self.sem_cat_vocab})
        self.lf_shell_to_sem_probs = self.shell_meaning_learner.all_inverse_distributions(self.shell_lf_vocab,base)

        self.word_to_sem_probs = self.word_to_lf_probs[self.lf_to_lf_shell_probs.index].dot(self.lf_to_lf_shell_probs)[self.lf_shell_to_sem_probs.index].dot(self.lf_shell_to_sem_probs)
        self.word_to_sem_probs = self.word_to_sem_probs.astype(pd.SparseDtype('float',0))
        self.combinator_probs =pd.DataFrame([self.syntax_learner.inverse_distribution(m) for m in self.splits_vocab],index=self.splits_vocab)

    def leaf_probs_of_word_span(self,words,beam_size):
        try:
            lf_probs = self.word_to_lf_probs.loc[words]
        except KeyError: # words has never been observed as a leaf
            return []
        lf_lf_shell_probs = self.lf_to_lf_shell_probs.mul(lf_probs,axis='index') # joint dist.
        lf_sem_probs = lf_lf_shell_probs.dot(self.lf_shell_to_sem_probs)
        sem_to_syn = [(sem,syn) for sem in lf_sem_probs.columns for syn in self.syn_cat_vocab if is_direct_congruent(sem,syn)]
        lf_syn_probs = pd.DataFrame([lf_sem_probs[a] for a,b in sem_to_syn],index=[b for a,b in sem_to_syn]).T
        lf_syn_probs *= [self.syntax_learner.prob('leaf',cat) for cat in lf_syn_probs.columns]
        paths_to_remember = lf_syn_probs.idxmax(axis=0)
        probs = [lf_syn_probs.loc[yv,yk] for yk,yv in paths_to_remember.items()]
        sorted_paths_and_probs = sorted(zip(paths_to_remember.items(),probs),key=lambda x:x[1])
        options = [{'syn_cat':k,'lf':v,'backpointer':None,'rule':'leaf','prob':p,'words':words}
            for (k,v),p in sorted_paths_and_probs[-beam_size:] if p>0]
        return sorted(options,key=lambda x:x['prob'])[-beam_size:]

    def parse(self,words):
        N = len(words)
        beam_size = 5
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
                        lsyn_cat, rsyn_cat = left_option['syn_cat'], right_option['syn_cat']
                        should_type_raise, outcat = is_fit_by_type_raise(lsyn_cat,rsyn_cat)
                        if should_type_raise:
                            lsyn_cat = type_raise(lsyn_cat,'fwd',outcat)
                        combined,rule = get_combination(lsyn_cat,rsyn_cat)
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
                        split = lsyn_cat + ' + ' + rsyn_cat
                        prob = left_option['prob']*right_option['prob']*self.syntax_learner.prob(split,combined)
                        backpointer = (k,j,left_idx), (i-k,j+k,right_idx)
                        #backpointer contains the coordinates in probs_table, and the idx in the
                        #beam, of the two locations that the current one could be split into
                        pn = {'sem_cat':combined,'syn_cat':combined,'lf':lf,'backpointer':backpointer,'rule':rule,'prob':prob,'words':words[j:j+i]}
                        possible_nexts.append(pn)
            if i == N:
                for pn in possible_nexts:
                    pn['prob'] = pn['prob']*self.root_sem_cat_memory.prob(pn['syn_cat'])
            to_add =sorted(possible_nexts,key=lambda x:x['prob'])[-beam_size:]
            probs_table[i-1,j] = to_add

        for a in range(2,N+1):
            for b in range(N-a+1):
                add_prob_of_span(a,b)
        if len(probs_table[N-1,0]) == 0:
            return 'No parse found'
        frontier = [probs_table[N-1,0][-1]]

        all_frontiers = []
        for i in range(N):
            new_frontier = []
            for item in frontier:
                if item['rule'] == 'leaf':
                    assert item['backpointer'] is None
                    new_frontier.append(item)
                    continue
                backpointer = item['backpointer']
                (left_len,left_pos,left_idx), (right_len,right_pos,right_idx) = backpointer
                left_split = probs_table[left_len-1,left_pos][left_idx]
                right_split = probs_table[right_len-1,right_pos][right_idx]
                new_frontier.append(left_split)
                new_frontier.append(right_split)
            all_frontiers.append(frontier)
            if all([x['rule']=='leaf' for x in frontier]): # parse is already at all leaves
                break
            frontier = copy(new_frontier)
        if ARGS.db_parse:
            print(probs_table)
            pprint(all_frontiers)
            breakpoint()
        #cmp_derivs = [x for x in probs_table[-1][0] if x['rule'] == 'fwd_cmp']
        #if len(cmp_derivs) > 0:
            #assert len(cmp_derivs) == 1
            #print(cmp_derivs[0])
        return all_frontiers[-1]

    def prob_of_split(self,x,is_map): # slow, just use for pdb
        if x['left'].is_fwd:
            f = x['left'].syn_cat
            g = x['right'].syn_cat
        else:
            f = x['right'].syn_cat
            g = x['left'].syn_cat
        if '\\' in g or '/' in g:
            parent_syn_cat = f[:-len(g)+3] # brackets
        else:
            parent_syn_cat = f[:-len(g)+1]
        split_prob = self.syntax_learner.prob(g + ' + ' + f,parent_syn_cat)
        prob_cache = {}
        left_prob = x['left'].probs(self.syntax_learner,self.shell_meaning_learner,
                    self.meaning_learner,self.word_learner,prob_cache,split_prob=1,is_map=is_map)
        right_prob = x['right'].probs(self.syntax_learner,self.shell_meaning_learner,
                    self.meaning_learner,self.word_learner,prob_cache,split_prob=1,is_map=is_map)
        return split_prob*left_prob*right_prob

if __name__ == "__main__":
    ARGS = argparse.ArgumentParser()
    ARGS.add_argument("--expname", type=str,default='tmp')
    ARGS.add_argument("--reload_from", type=str)
    ARGS.add_argument("--num_dpoints", type=int, default=-1)
    ARGS.add_argument("--db_at", type=int, default=-1)
    ARGS.add_argument("--max_sent_len", type=int, default=9)
    ARGS.add_argument("--num_epochs", type=int, default=1)
    ARGS.add_argument("--num_generate", type=int, default=0)
    ARGS.add_argument("-tt","--short_test_run", action="store_true")
    ARGS.add_argument("-t","--test_run", action="store_true")
    ARGS.add_argument("--show_splits", action="store_true")
    ARGS.add_argument("--db_parse", action="store_true")
    ARGS.add_argument("--db_after", action="store_true")
    ARGS.add_argument("--overwrite", action="store_true")
    ARGS.add_argument("--shuffle", action="store_true")
    ARGS.add_argument("-d", "--dset", type=str, default='determiners1000')
    ARGS.add_argument("--cat_to_sample_from", type=str, default='S')
    ARGS = ARGS.parse_args()

    ARGS.test_run = ARGS.test_run or ARGS.short_test_run

    if ARGS.test_run:
        ARGS.expname = 'tmp'
        if ARGS.num_dpoints == -1:
            ARGS.num_dpoints = 10
    elif ARGS.expname is None:
        expname = f'{ARGS.num_epochs}_{ARGS.max_sent_len}'
    else:
        expname = ARGS.expname
    if ARGS.short_test_run:
        ARGS.num_dpoints = 30
    set_experiment_dir(f'experiments/{ARGS.expname}',overwrite=ARGS.overwrite,name_of_trials='experiments/tmp')
    with open(f'data/{ARGS.dset}.json') as f: d=json.load(f)

    NAMES = d['np_list'] + [str(x) for x in range(1,11)] # because of the numbers in simple dsets
    NOUNS = d['nouns']
    TRANSITIVES = d['transitive_verbs']
    INTRANSITIVES = d['intransitive_verbs']
    base_lexicon = {w:cat for item,cat in zip([NAMES,NOUNS,INTRANSITIVES,TRANSITIVES],('NP','N','S|NP','S|NP|NP')) for w in item}

    language_acquirer = LanguageAcquirer(base_lexicon)
    if ARGS.reload_from is not None:
        language_acquirer.load_from(f'experiments/{ARGS.reload_from}')
    if ARGS.shuffle: np.random.shuffle(d['data'])
    NDPS = len(d['data']) if ARGS.num_dpoints == -1 else ARGS.num_dpoints
    f = open(f'experiments/{ARGS.expname}/results.txt','w')
    all_data = d['data'][:NDPS]
    train_data = all_data[:-len(all_data)//5]
    test_data = all_data[-len(all_data)//5:]
    start_time = time()
    for epoch_num in range(ARGS.num_epochs):
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
        num_correct_parses = 0

    inverse_probs_start_time = time()
    language_acquirer.compute_inverse_probs()
    print(f'Time to compute inverse probs: {time()-inverse_probs_start_time:.3f}s')
    meaning_acc ,syn_acc = language_acquirer.test_NPs()
    file_print(f'Accuracy at meaning of state names: {meaning_acc:.1f}%',f)
    file_print(f'Accuracy at syn-cat of state names: {syn_acc:.1f}%',f)
    file_print(f'\nSamples for type {ARGS.cat_to_sample_from}:',f)
    for _ in range(ARGS.num_generate):
        generated = language_acquirer.generate_words(ARGS.cat_to_sample_from)
        if any([x['words'][:-1]==generated.split() for x in d['data']]):
            file_print(f'{generated}: seen during training',f)
        else:
            file_print(f'{generated}: not seen during training',f)
    file_print(f'Total run time: {time()-start_time:.3f}s',f)
    f.close()
    for dpoint in d['data'][:10]:
        pprint(language_acquirer.parse(dpoint['words']))
    print(language_acquirer.syntax_learner.memory['S/NP'])
    if ARGS.db_after:
        breakpoint()
    language_acquirer.save_to(f'experiments/{ARGS.expname}')
