import numpy as np
from dl_utils.misc import set_experiment_dir
from os.path import join
import pandas as pd
from utils import normalize_dict, split_respecting_brackets, file_print, get_combination
from time import time
import argparse
from abc import ABC
import re
from parser import LogicalForm, ParseNode
import json


class BaseDirichletProcessLearner(ABC):
    def __init__(self,alpha):
        self.alpha = alpha
        self.memory = {}
        self.base_distribution_cache = {}

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

    def observe(self,y,x,weight):
        if x not in self.memory:
            self.memory[x] = {y:weight,'count':weight}
        elif y not in self.memory[x]:
            self.memory[x] = self.memory[x] | {y:weight}
            self.memory[x]['count'] += weight
        else:
            self.memory[x][y] += weight
            self.memory[x]['count'] += weight
        assert np.allclose(2*self.memory[x]['count'],sum(self.memory[x].values()),rtol=1e-6)

    def inverse_distribution(self,y): # conditional on y
        seen_before = any([y in d for d in self.memory.values()])
        assert seen_before, f'learner hasn\'t seen word \'{y}\' before'
        inverse_distribution = {x:m.get(y,0) for x,m in self.memory.items()}
        return normalize_dict(inverse_distribution)

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

class CCGDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        if x == 'NP + S/NP\\NP':
            breakpoint()
            return 0
        num_slashes = len(re.findall(r'[\\/\|]',x))
        return 0.2**(num_slashes+1)

class ShellMeaningDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        num_vars = len(set(re.findall(r'\$\d',x)))
        num_constants = float('PLACEHOLDER' in x)
        return np.e**(-2*num_vars - num_constants)

class MeaningDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        num_vars = len(set(re.findall(r'\$\d',x)))
        num_constants = len(set([z for z in re.findall(r'[a-z]*',x) if 'lambda' not in z and len(z)>0]))
        return np.e**(-2*num_vars - num_constants)

class WordSpanDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution(self,x):
        assert len(x) > 0
        return 28**(-len(x)+1)

class LanguageAcquirer():
    def __init__(self,base_lexicon):
        self.base_lexicon = base_lexicon
        self.syntax_learner = CCGDirichletProcessLearner(1)
        self.shell_meaning_learner = ShellMeaningDirichletProcessLearner(1000)
        self.meaning_learner = MeaningDirichletProcessLearner(500)
        self.word_learner = WordSpanDirichletProcessLearner(1)
        self.lf_cache = {} # maps lf_strings to LogicalForm objects
        self.lf_splits_cache = {} # maps LogicalForm objects to lists of (left-child,right-child)
        self.parse_node_cache = {} # maps utterances (str) to ParseNode objects, including splits

    @property
    def lf_vocab(self):
        return self.word_learner.memory.keys()

    @property
    def shell_lf_vocab(self):
        return self.meaning_learner.memory.keys()

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

        self.word_to_sem_probs = pd.read_pickle(join(fpath,'word_to_sem_probs.pkl'))
        self.sem_to_sem_shell_probs = pd.read_pickle(join(fpath,'sem_to_sem_shell_probs.pkl'))
        self.sem_shell_to_syn_probs = pd.read_pickle(join(fpath,'sem_shell_to_syn_probs.pkl'))
        self.word_to_syn_probs = pd.read_pickle(join(fpath,'word_to_syn_probs.pkl'))

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

        self.word_to_sem_probs.to_pickle(join(fpath,'word_to_sem_probs.pkl'))
        self.sem_to_sem_shell_probs.to_pickle(join(fpath,'sem_to_sem_shell.pkl'))
        self.sem_shell_to_syn_probs.to_pickle(join(fpath,'sem_shell_to_syn_probs.pkl'))
        self.word_to_syn_probs.to_pickle(join(fpath,'word_to_syn_probs.pkl'))

    def show_word_meanings(self,word): # prob of meaning given word assuming flat prior over meanings
        distr = self.word_learner.inverse_distribution(word)
        probs = sorted(distr.items(), key=lambda x:x[1])[-15:]
        print(f'\nLearned Meaning for \'{word}\'')
        print('\n'.join([f'{prob:.3f}: {word}' for word,prob in probs]))

    def compute_inverse_probs(self): # prob of meaning given word assuming flat prior over meanings
        self.word_to_sem_probs = pd.DataFrame([self.word_learner.inverse_distribution(w) for w in self.mwe_vocab],index=self.mwe_vocab)
        self.word_to_sem_probs *= pd.Series(self.meaning_learner.base_distribution_cache)
        self.word_to_sem_probs = self.word_to_sem_probs.astype(pd.SparseDtype('float',0))

        self.sem_to_sem_shell_probs = pd.DataFrame([self.meaning_learner.inverse_distribution(m) for m in self.lf_vocab],index=self.lf_vocab)
        self.sem_to_sem_shell_probs *= pd.Series(self.shell_meaning_learner.base_distribution_cache)
        self.sem_to_sem_shell_probs = self.sem_to_sem_shell_probs.astype(pd.SparseDtype('float',0))

        self.sem_shell_to_syn_probs = pd.DataFrame([self.shell_meaning_learner.inverse_distribution(m) for m in self.shell_lf_vocab],index=self.shell_lf_vocab)
        self.sem_shell_to_syn_probs *= pd.Series({x:self.syntax_learner.base_distribution(x) for x in self.sem_shell_to_syn_probs.columns})
        self.sem_shell_to_syn_probs = self.sem_shell_to_syn_probs.astype(pd.SparseDtype('float',0))

        self.word_to_syn_probs = self.word_to_sem_probs.dot(self.sem_to_sem_shell_probs).dot(self.sem_shell_to_syn_probs)
        self.word_to_syn_probs = self.word_to_syn_probs.astype(pd.SparseDtype('float',0))
        self.combinator_probs =pd.DataFrame([self.syntax_learner.inverse_distribution(m) for m in self.splits_vocab],index=self.splits_vocab)

    def show_word(self,word,f=None):
        meanings = self.word_to_sem_probs.loc[word].sort_values()[-10:]
        file_print(f'\nLearned meanings for \'{word}\'',f)
        file_print('\n'.join([f'{word}: {100*prob:.2f}%' for word,prob in meanings.items() if prob > 1e-4]),f)

        syn_cats = self.word_to_syn_probs.loc[word].sort_values()[-10:]
        file_print(f'\nLearned syntactic categories for \'{word}\'',f)
        file_print('\n'.join([f'{word}: {100*prob:.2f}%' for word,prob in syn_cats.items() if prob > 1e-4]),f)

    def show_splits(self,syn_cat,f):
        file_print(f'Learned splits for category {syn_cat}',f)
        counts = self.syntax_learner.memory[syn_cat]
        norm = counts['count']
        probs = {k:counts[k]/norm for k in sorted(counts,key=lambda x:counts[x]) if k!='count'}
        file_print('\n'.join([f'{prob:.3f}: {word}' for word,prob in probs.items()]),f)

    def get_lf(self,lf_str):
        if lf_str in self.lf_cache:
            return self.lf_cache[lf_str]
        else:
            lf = LogicalForm(lf_str,self.base_lexicon,self.lf_splits_cache,parent='START',sem_cat=ARGS.root_sem_cat)
            self.lf_cache[lf_str] = lf
            return lf

    def train_one_step(self,lf_str,words):
        parse_root = self.make_parse_node(lf_str,words)
        prob_cache = {}
        root_prob = parse_root.probs(self.syntax_learner,self.shell_meaning_learner,
                       self.meaning_learner,self.word_learner,prob_cache,split_prob=1,is_map=False)
        parse_root.propagate_down_prob(1)
        for node, prob in prob_cache.items():
            if node.parent is not None and not node.is_g:
                if node.is_fwd:
                    syntax_split = node.syn_cat + ' + ' + node.sibling.syn_cat
                else:
                    syntax_split = node.sibling.syn_cat + ' + ' + node.syn_cat
                update_weight = node.subtree_prob * node.down_prob / root_prob # for conditional
                self.syntax_learner.observe(syntax_split,node.parent.syn_cat,weight=update_weight)
            leaf_prob = node.down_prob*node.stored_prob_as_leaf/root_prob
            shell_lf = node.logical_form.subtree_string(as_shell=True,alpha_normalized=True)
            lf = node.logical_form.subtree_string(alpha_normalized=True)
            word_str = ' '.join(node.words)
            self.syntax_learner.observe('leaf',node.syn_cat,weight=leaf_prob)
            self.shell_meaning_learner.observe(shell_lf,node.sem_cat,weight=leaf_prob)
            self.meaning_learner.observe(lf,shell_lf,weight=leaf_prob)
            self.word_learner.observe(word_str,lf,weight=leaf_prob)

    def test_NPs(self):
        meaning_corrects = 0
        syn_corrects = 0
        attempts = 0
        nps = [x for x,c in self.base_lexicon.items() if c=='NP']
        for w in nps:
            w_spaces = w.replace('_',' ')
            if w_spaces not in self.word_to_sem_probs.index:
                if w_spaces not in ['virginia','kansas','montgomery']:
                    if not not any([w_spaces in ' '.join(x['words']) and len([z for z in x['words'] if z != '?']) <= ARGS.max_sent_len for x in d['data'][:NDPS]]):
                        print(f'\'{w_spaces}\' not here')
                continue
            meaning_pred = self.word_to_sem_probs.loc[w_spaces].idxmax()
            syn_pred = self.word_to_syn_probs.loc[w_spaces].idxmax()
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
            self.parse_node_cache[' '.join([lf_str]+words)] = parse_root
        return parse_root

    def MAP_analysis(self,lf_str,words):
        parse_root = self.make_parse_node(lf_str,words)
        prob_cache = {}
        root_prob = parse_root.probs(self.syntax_learner,self.shell_meaning_learner,
                       self.meaning_learner,self.word_learner,prob_cache,split_prob=1,is_map=True)

        frontier = [parse_root]
        layers = []
        while True:
            if frontier != []:
                layers.append(frontier)
            if all([x.is_leaf for x in frontier]):
                break
            new_frontier = []
            for node in frontier:
                if len(node.possible_splits) > 0:
                    assert all([x['left'].split_prob == x['right'].split_prob for x in node.possible_splits])
                    y = max(node.possible_splits, key=lambda x: x['left'].subtree_prob*x['right'].subtree_prob*x['left'].split_prob)
                    if y['left'].subtree_prob*y['right'].subtree_prob*y['left'].split_prob > node.stored_prob_as_leaf:
                        new_frontier += [y['left'],y['right']]
            frontier = new_frontier
        return layers

    def possible_next_frontiers(self,syn_cats):
        possible_frontiers = []
        for i in range(len(syn_cats)-1):
            maybe_left, maybe_right = syn_cats[i], syn_cats[i+1]
            combination = get_combination(maybe_left,maybe_right)
            if combination is not None:
                split_prob = self.syntax_learner.prob(f'{maybe_left} + {maybe_right}',combination)
                possible_frontiers.append((syn_cats[:i]+[combination]+syn_cats[i+2:],split_prob))
        return possible_frontiers

    def parses_of_syn_cats(self,syn_cats):
        """Return all possible parses (often only one) of the given syn_cats in the given order."""
        frontiers = [([syn_cats],1)]
        for _ in range(len(syn_cats)-1):
            frontiers = [(cur[0]+[f[0]],cur[1]*f[1]) for cur in frontiers for f in self.possible_next_frontiers(cur[0][-1])]
        return frontiers

    def parse(self,words):
        N = len(words)
        beam_size = 5
        probs_table = np.empty((N,N),dtype='object') #(i,j) will be a dict of len(beam_size) saying probs of top syn_cats
        for i in range(N):
            sem_probs = self.word_to_sem_probs.loc[words[i]]
            sem_sem_shell_probs = self.sem_to_sem_shell_probs.mul(sem_probs,axis='index')
            sem_syn_probs = sem_sem_shell_probs.dot(self.sem_shell_to_syn_probs)
            paths_to_remember = sem_syn_probs.idxmax(axis=0)
            probs = [sem_syn_probs.loc[yv,yk] for yk,yv in paths_to_remember.items()]
            probs_table[0,i] = sorted(zip(paths_to_remember.items(),probs),key=lambda x: x[1])[-beam_size:]

        def add_prob_of_span(i,j):
            for k in range(1,i):
                left_chunk_probs = probs_table[i-k-1,j]
                right_chunk_probs = probs_table[k-1,j+k] # so total len is always j
                for (lsem_cat,llf), lprob in left_chunk_probs:
                    for (rsem_cat,rlf), rprob in right_chunk_probs:
                        combined,rule = get_combination(lsem_cat,rsem_cat)
                        if combined:
                            breakpoint()

                # now consider the different ways to combine them
        for a in range(2,N):
            for b in range(N-a):
                add_prob_of_span(a,b)

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
    ARGS.add_argument("--expname", type=str)
    ARGS.add_argument("--reload_from", type=str)
    ARGS.add_argument("--num_dpoints", type=int, default=-1)
    ARGS.add_argument("--db_at", type=int, default=-1)
    ARGS.add_argument("--max_sent_len", type=int, default=6)
    ARGS.add_argument("--num_epochs", type=int, default=1)
    ARGS.add_argument("--devel", "--development_mode", action="store_true")
    ARGS.add_argument("--show_splits", action="store_true")
    ARGS.add_argument("--breakin", action="store_true")
    ARGS.add_argument("--overwrite", action="store_true")
    ARGS.add_argument("--shuffle", action="store_true")
    ARGS.add_argument("-d", "--dset", type=str, default='simple1000')
    ARGS.add_argument("--root_sem_cat", type=str, default='S')
    ARGS = ARGS.parse_args()

    if ARGS.expname is None:
        expname = f'{ARGS.num_epochs}_{ARGS.max_sent_len}_root-{ARGS.root_sem_cat}'
    else:
        expname = ARGS.expname
    set_experiment_dir(f'experiments/{expname}',overwrite=ARGS.overwrite,name_of_trials='experiments/tmp')
    with open(f'data/{ARGS.dset}.json') as f: d=json.load(f)

    NPS = d['np_list'] + [str(x) for x in range(1,11)] # because of the numbers in simple dsets
    TRANSITIVES = d['transitive_verbs']
    INTRANSITIVES = d['intransitive_verbs']
    base_lexicon = {w:cat for item,cat in zip([NPS,INTRANSITIVES,TRANSITIVES],('NP','S|NP','S|NP|NP')) for w in item}

    language_acquirer = LanguageAcquirer(base_lexicon)
    if ARGS.reload_from is not None:
        language_acquirer.load_from(f'experiments/{ARGS.reload_from}')
    if ARGS.shuffle: np.random.shuffle(d['data'])
    NDPS = len(d['data']) if ARGS.num_dpoints == -1 else ARGS.num_dpoints
    f = open(f'experiments/{expname}/results.txt','w')
    start_time = time()
    for epoch_num in range(ARGS.num_epochs):
        epoch_start_time = time()
        for i,dpoint in enumerate(d['data'][:NDPS]):
            if i < 10 and epoch_num > 0:
                continue
            words, lf_str = dpoint['words'], dpoint['parse']
            if words[-1] == '?':
                words = words[:-1]
            if i == ARGS.db_at:
                breakpoint()
            if len(words) <= ARGS.max_sent_len:
                language_acquirer.train_one_step(lf_str,words)
        print(f"Epoch {epoch_num} completed, time taken: {time()-epoch_start_time:.3f}s")
        language_acquirer.show_splits(ARGS.root_sem_cat,f)
        final_parses = {}
        num_correct_parses = 0
        for dpoint in d['data']:
            favoured_parse = language_acquirer.MAP_analysis(dpoint['parse'],dpoint['words'])
            final_syn_cats = ' + '.join([x.syn_cat for x in favoured_parse[-1]])
            if final_syn_cats == 'NP':
                breakpoint()
            try:
                final_parses[final_syn_cats] += 1
            except KeyError:
                final_parses[final_syn_cats] = 1
            if final_syn_cats == 'S\\NP/NP + NP' and len(dpoint['parse'].split())==3:
                num_correct_parses += 1
            elif final_syn_cats == 'NP + S\\NP' and len(dpoint['parse'].split())==2:
                num_correct_parses += 1
        print(final_parses)
        print(f"parse accuracy: {num_correct_parses/len(d['data']):.3f}")

    inverse_probs_start_time = time()
    language_acquirer.compute_inverse_probs()
    print(f'Time to compute inverse probs: {time()-inverse_probs_start_time:.3f}s')
    meaning_acc ,syn_acc = language_acquirer.test_NPs()
    file_print(f'Accuracy at meaning of state names: {meaning_acc:.1f}%',f)
    file_print(f'Accuracy at syn-cat of state names: {syn_acc:.1f}%',f)
    file_print(f'\nSamples for type {ARGS.root_sem_cat}:',f)
    for _ in range(10):
        generated = language_acquirer.generate_words(ARGS.root_sem_cat)
        if any([x['words'][:-1]==generated.split() for x in d['data']]):
            file_print(f'{generated}: seen during training',f)
        else:
            file_print(f'{generated}: not seen during training',f)
    file_print(f'Total run time: {time()-start_time:.3f}s',f)
    f.close()
    print(language_acquirer.parse(d['data'][0]['words']))
    if ARGS.breakin:
        breakpoint()
    language_acquirer.save_to(f'experiments/{expname}')
