import numpy as np
import pandas as pd
from utils import normalize_dict
from time import time
from pprint import pprint
import argparse
from abc import ABC
import re
from easy_split import LogicalForm, ParseNode
import json


class SimpleDirichletProcess(ABC):
    def __init__(self,alpha):
        self.alpha = alpha
        self.memory = {}
        self.count = 0

    def base_distribution(self,x):
        raise NotImplementedError

    def prob(self,x):
        mx = self.memory.get(x,0)
        base = self.base_distribution(x)
        return (mx+self.alpha*base)/(self.count+self.alpha)

    def observe(self,x,weight):
        if x in self.memory:
            self.memory[x] += weight
        else:
            self.memory[x] = weight
        self.count += weight

    def show(self):
        scores = sorted(self.memory.items(), key=lambda x: x[1])
        scores = normalize_dict(scores)
        pprint(scores[-25:])

class BaseDirichletProcessLearner(ABC):
    def __init__(self,alpha):
        self.alpha = alpha
        self.memory = pd.DataFrame([],columns=['count'])
        self.base_distribution_cache = pd.Series([],dtype=float)
        self.x_vocab = []
        self.y_vocab = []

    def count(self,x):
        return self.memory.loc[x,'count']

    def base_distribution_(self,x):
        raise NotImplementedError

    def prob(self,y,x): # prob of y given x
        base_prob = self.base_distribution(y)
        if x not in self.memory.index:
            return base_prob
        mem_count = self.memory.loc[x].get('count',0)
        mem_count_y = self.memory.loc[x].get(y,0)
        if mem_count_y!=mem_count_y:
            return base_prob # it's nan, which means 0
        p = (mem_count_y + self.alpha*base_prob)/(mem_count+self.alpha)
        if p!=p:
            breakpoint()
        return p

    def observe(self,y,x,weight):
        #if isinstance(self,CCGDirichletProcess):
            #breakpoint()
        if x not in self.x_vocab: self.x_vocab.append(x)
        if y not in self.y_vocab: self.y_vocab.append(y)
        if x in self.memory.index and y in self.memory.columns and self.memory.loc[x].notna().loc[y]:
            self.memory.loc[x,y] += weight
            self.memory.loc[x,'count'] += weight
        else:
            self.memory.loc[x,y] = weight
            if self.memory.loc[x,'count']!=self.memory.loc[x,'count']:
                self.memory.loc[x,'count'] = weight
            else:
                self.memory.loc[x,'count'] += weight
        assert (self.memory.sum(axis=1) - 2*self.memory['count']).max() < 1e-10

    def inverse_distribution(self,y): # conditional on y
        seen_before = y in self.memory.columns
        assert seen_before, f'learner hasn\'t seen word \'{y}\' before'
        inverse_distribution = {x:self.prob(y,x)*self.count(x) for x,d in self.memory.index}
        return normalize_dict(inverse_distribution)

    def base_distribution(self,x):
        if x in self.base_distribution_cache.index:
            return self.base_distribution_cache.loc[x]
        base_prob = self.base_distribution_(x)
        self.base_distribution_cache[x] = base_prob
        return base_prob

    def prob_mat(self):
        return (self.memory.drop('count') + self.base_distribution_cache*self.alpha)/(self.memory['count']+self.alpha)

    def inverse_prob_mat(self):
        unnormed_probs = self.memory.drop('count',axis=1) + self.base_distribution_cache*self.alpha
        normed_probs = unnormed_probs/unnormed_probs.sum(axis=0)
        return normed_probs.T

class CCGDirichletProcess(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        num_slashes = len(re.findall(r'\\/\|',x))
        return 0.2**(num_slashes+1)

class ShellMeaningDirichletProcess(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        num_vars = len(set(re.findall(r'\$\d',x)))
        num_constants = len(set([z for z in re.findall(r'[a-z]*',x) if 'lambda' not in z and len(z)>0]))
        return np.e**(2*num_vars + num_constants)

class MeaningDirichletProcess(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        return 1 # unnormalized uniform, hope this works, may end up unfairly rewarding depth

class WordSpanDirichletProcess(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        assert len(x) > 0
        return 1/27 * 28**(-len(x)+1)

class LanguageAcquirer():
    def __init__(self,base_lexicon,syntax_alpha,shell_meaning_alpha,meaning_alpha,word_alpha):
        self.base_lexicon = base_lexicon
        self.syntax_alpha = syntax_alpha
        self.shell_meaning_alpha = shell_meaning_alpha
        self.meaning_alpha = meaning_alpha
        self.word_alpha = word_alpha
        self.syntax_learner = CCGDirichletProcess(syntax_alpha)
        self.shell_meaning_learner = ShellMeaningDirichletProcess(shell_meaning_alpha)
        self.meaning_learner = MeaningDirichletProcess(meaning_alpha)
        self.word_learner = WordSpanDirichletProcess(word_alpha)
        self.lf_splits_cache = {}

    @property
    def lf_vocab(self):
        return self.word_learner.distributions.keys()

    @property
    def shell_lf_vocab(self):
        return self.meaning_learner.distributions.keys()

    @property
    def mwe_vocab(self):
        return set([w for x in self.word_learner.distributions.values() for w in x.memory.keys()])

    @property
    def vocab(self):
        return list(set([w for x in self.word_learner.distributions.values()
                        for w in x.memory.keys() if ' ' not in w]))

    def show_word_meanings(self,word): # prob of meaning given word assuming flat prior over meanings
        distr = self.word_learner.inverse_distribution(word)
        probs = sorted(distr.items(), key=lambda x:x[1])[-15:]
        print(f'\nLearned Meaning for \'{word}\'')
        print('\n'.join([f'{prob:.3f}: {word}' for word,prob in probs]))

    def compute_inverse_probs(self): # prob of meaning given word assuming flat prior over meanings
        self.word_to_sem_probs = self.word_learner.inverse_prob_mat()
        self.sem_to_sem_shell_probs = self.meaning_learner.inverse_prob_mat()
        self.sem_shell_to_syn_probs = self.shell_meaning_learner.inverse_prob_mat()
        self.word_to_syn_probs = self.word_to_sem_probs.dot(self.sem_to_sem_shell_probs).dot(self.sem_shell_to_syn_probs)

    def show_word(self,word):
        with pd.option_context('display.float_format', '${:,.2f}'.format):
            breakpoint()
            print(self.word_to_sem_probs.loc[word].sort_values()[-10:].view())
            print(self.word_to_syn_probs.loc[word].sort_values()[-10:].view())

    def show_splits(self,syn_cat):
        if syn_cat not in self.syntax_learner.memory.index:
            print(f'learner hasn\'t seen word \'{syn_cat}\' before')
        probs = self.syntax_learner.memory.loc[syn_cat]/self.syntax_learner.count(syn_cat)
        #unnormed_counts = self.syntax_learner.memory[syn_cat].memory
        #probs = normalize_dict(unnormed_counts)
        #probs = {k:probs[k] for k in sorted(probs,key=lambda x:probs[x])}
        print('\n'.join([f'{prob:.3f}: {word}' for word,prob in probs.items()]))

    def train_one_step(self,words,logical_form_str):
        start_time = time()
        lf = LogicalForm(logical_form_str,self.base_lexicon,self.lf_splits_cache,parent='START')
        parse_root = ParseNode(lf,words,'ROOT')
        prob_cache = {}
        root_prob = parse_root.prob(self.syntax_learner,
                self.shell_meaning_learner,self.meaning_learner,self.word_learner,prob_cache)
        for node, prob in prob_cache.items():
            if node.parent is not None and not node.is_g:
                syntax_split = node.syn_cat + ' + ' + node.sibling.syn_cat
                update_weight = node.stored_prob * node.sibling.stored_prob # prob data given split
                self.syntax_learner.observe(syntax_split,node.parent.syn_cat,weight=update_weight)
            shell_lf = node.logical_form.subtree_string(as_shell=True,alpha_normalized=True)
            lf = node.logical_form.subtree_string(alpha_normalized=True)
            word_str = ' '.join(node.words)
            self.shell_meaning_learner.observe(shell_lf,node.syn_cat,weight=prob)
            self.meaning_learner.observe(lf,shell_lf,weight=prob)
            self.word_learner.observe(word_str,lf,weight=prob)
        print(f'{time()-start_time:.4f}')


if __name__ == "__main__":
    ARGS = argparse.ArgumentParser()
    ARGS.add_argument("--expname", type=str, default='tmp',
                          help="the directory to write output files")
    ARGS.add_argument("--dset", type=str, choices=['easy-adam','geo'], default="geo")
    ARGS.add_argument("--num_dpoints", type=int, default=-1)
    ARGS.add_argument("--db_at", type=int, default=-1)
    ARGS.add_argument("--max_sent_len", type=int, default=6)
    ARGS.add_argument("--num_epochs", type=int, default=1)
    ARGS.add_argument("--is_dump_verb_repo", action="store_true",
                          help="whether to dump the verb repository")
    ARGS.add_argument("--devel", "--development_mode", action="store_true")
    ARGS.add_argument("--show_splits", action="store_true")
    ARGS.add_argument("--simple_example", action="store_true")
    ARGS = ARGS.parse_args()

    with open('data/preprocessed_geoqueries.json') as f: d=json.load(f)

    NPS = d['np_list']
    TRANSITIVES = d['transitive_verbs']
    INTRANSITIVES = d['intransitive_verbs']
    base_lexicon = {w:cat for item,cat in zip([NPS,INTRANSITIVES,TRANSITIVES],('NP','S|NP','S|NP|NP')) for w in item}

    language_acquirer = LanguageAcquirer(base_lexicon,1,1,1,1)
    ndps = len(d['data']) if ARGS.num_dpoints == -1 else ARGS.num_dpoints
    start_time = time()
    for epoch_num in range(ARGS.num_epochs):
        for i,dpoint in enumerate(d['data'][:ndps]):
            if ARGS.simple_example:
                lf_str = 'loc colorado virginia'
                words = ['colarado', 'is', 'in', 'virginia']
            else:
                words, lf_str = dpoint['words'], dpoint['parse']
            if words[-1] == '?':
                words = words[:-1]
            if i == ARGS.db_at:
                breakpoint()
            if len(words) > ARGS.max_sent_len:
                print(f"excluding because too long: {' '.join(words)}")
                continue
            language_acquirer.train_one_step(words,lf_str)
    language_acquirer.show_splits('S')
    language_acquirer.compute_inverse_probs()
    language_acquirer.show_word('virginia')
    language_acquirer.show_word('cities')
    print(f'Total run time: {time()-start_time:.3f}s')
