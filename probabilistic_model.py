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


class BaseDirichletProcessLearner(ABC):
    def __init__(self,alpha):
        self.alpha = alpha
        self.memory = {}
        self.base_distribution_cache = {}

    def base_distribution_(self,x):
        raise NotImplementedError

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
        assert 2*self.memory[x]['count'] == sum(self.memory[x].values())

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

class CCGDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        num_slashes = len(re.findall(r'\\/\|',x))
        return 0.2**(num_slashes+1)

class ShellMeaningDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        num_vars = len(set(re.findall(r'\$\d',x)))
        num_constants = len(set([z for z in re.findall(r'[a-z]*',x) if 'lambda' not in z and len(z)>0]))
        return np.e**(-2*num_vars - num_constants)

class MeaningDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution_(self,x):
        num_vars = len(set(re.findall(r'\$\d',x)))
        num_constants = float('PLACEHOLDER' in x)
        return np.e**(-2*num_vars - num_constants)

class WordSpanDirichletProcessLearner(BaseDirichletProcessLearner):
    def base_distribution(self,x):
        assert len(x) > 0
        return 1/27 * 28**(-len(x)+1)

class LanguageAcquirer():
    def __init__(self,base_lexicon):
        self.base_lexicon = base_lexicon
        self.syntax_learner = CCGDirichletProcessLearner(1)
        self.shell_meaning_learner = ShellMeaningDirichletProcessLearner(1000)
        self.meaning_learner = MeaningDirichletProcessLearner(500)
        self.word_learner = WordSpanDirichletProcessLearner(1)
        self.lf_cache = {}
        self.lf_splits_cache = {}
        self.parse_node_cache = {}

    @property
    def lf_vocab(self):
        return self.word_learner.memory.keys()

    @property
    def shell_lf_vocab(self):
        return self.meaning_learner.memory.keys()

    @property
    def mwe_vocab(self):
        return set([w for x in self.word_learner.memory.values() for w in x.keys()])

    @property
    def vocab(self):
        return list(set([w for x in self.word_learner.memory.values()
                        for w in x.keys() if ' ' not in w]))

    def show_word_meanings(self,word): # prob of meaning given word assuming flat prior over meanings
        distr = self.word_learner.inverse_distribution(word)
        probs = sorted(distr.items(), key=lambda x:x[1])[-15:]
        print(f'\nLearned Meaning for \'{word}\'')
        print('\n'.join([f'{prob:.3f}: {word}' for word,prob in probs]))

    def compute_inverse_probs(self): # prob of meaning given word assuming flat prior over meanings
        start_time = time()
        self.word_to_sem_probs = pd.DataFrame([self.word_learner.inverse_distribution(w) for w in self.vocab],index=self.vocab)
        self.word_to_sem_probs *= pd.Series(self.meaning_learner.base_distribution_cache)
        self.sem_to_sem_shell_probs = pd.DataFrame([self.meaning_learner.inverse_distribution(m) for m in self.lf_vocab],index=self.lf_vocab)
        self.sem_to_sem_shell_probs *= pd.Series(self.shell_meaning_learner.base_distribution_cache)
        self.sem_shell_to_syn_probs = pd.DataFrame([self.shell_meaning_learner.inverse_distribution(m) for m in self.shell_lf_vocab],index=self.shell_lf_vocab)
        self.sem_shell_to_syn_probs *= pd.Series({x:self.syntax_learner.base_distribution(x) for x in self.sem_shell_to_syn_probs.columns})
        self.word_to_syn_probs = self.word_to_sem_probs.dot(self.sem_to_sem_shell_probs).dot(self.sem_shell_to_syn_probs)
        print(f"inverse_prob_time: {time()-start_time:.3f}")

    def show_word(self,word):
        with pd.option_context('display.float_format', '${:,.2f}'.format):
            print(self.word_to_sem_probs.loc[word].sort_values()[-10:].view())
            print(self.word_to_syn_probs.loc[word].sort_values()[-10:].view())

    def show_splits(self,syn_cat):
        counts = self.syntax_learner.memory[syn_cat]
        norm = counts['count']
        probs = {k:counts[k]/norm for k in sorted(counts,key=lambda x:counts[x]) if k!='count'}
        print('\n'.join([f'{prob:.3f}: {word}' for word,prob in probs.items()]))

    def train_one_step(self,words,logical_form_str):
        if logical_form_str in self.lf_cache:
            lf = self.lf_cache[logical_form_str]
        else:
            lf = LogicalForm(logical_form_str,self.base_lexicon,self.lf_splits_cache,parent='START')
            self.lf_cache[logical_form_str] = lf
        if ' '.join([logical_form_str]+words) in self.parse_node_cache:
            parse_root = self.parse_node_cache[' '.join([logical_form_str]+words)]
        else:
            parse_root = ParseNode(lf,words,'ROOT')
            self.parse_node_cache[' '.join([logical_form_str]+words)] = parse_root
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

    language_acquirer = LanguageAcquirer(base_lexicon)
    ndps = len(d['data']) if ARGS.num_dpoints == -1 else ARGS.num_dpoints
    start_time = time()
    for epoch_num in range(ARGS.num_epochs):
        print(f"Epoch: {epoch_num}")
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
                continue
            language_acquirer.train_one_step(words,lf_str)
    language_acquirer.show_splits('S')
    language_acquirer.compute_inverse_probs()
    language_acquirer.show_word('virginia')
    language_acquirer.show_word('cities')
    print(f'Total run time: {time()-start_time:.3f}s')
