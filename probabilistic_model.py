import numpy as np
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

class CCGDirichletProcess(SimpleDirichletProcess):
    def base_distribution(self,x):
        num_slashes = len(re.findall(r'\\/\|',x))
        return 0.2**(num_slashes+1)

class ShellMeaningDirichletProcess(SimpleDirichletProcess):
    def base_distribution(self,x):
        num_vars = len(x.var_descendents)
        num_constants = len([z for z in x.descendents if z.node_type in ('bound_var','unbound_var')])
        return np.e**(2*num_vars + num_constants)

class MeaningDirichletProcess(SimpleDirichletProcess):
    def base_distribution(self,x):
        return 1 # unnormalized uniform, hope this works, may end up unfairly rewarding depth

class WordSpanDirichletProcess(SimpleDirichletProcess):
    def base_distribution(self,x):
        #return 1/500**len(x)
        assert len(x) > 0
        return 1/27 * 28**(-len(x)+1)

class Learner():
    def __init__(self,alpha,dp_type):
        self.alpha = alpha
        self.distributions = {}
        self.dp_type = dp_type # type of Dirichlet process

    def prob(self,y,x): # prob of y given x
        if x in self.distributions:
            distribution = self.distributions[x]
        else:
            distribution = self.dp_type(self.alpha)
        return distribution.prob(y)

    def observe(self,y,x,weight):
        if x not in self.distributions:
            self.distributions[x] = self.dp_type(self.alpha)
        self.distributions[x].observe(y,weight)

    def inverse_distribution(self,y): # conditional on y
        inverse_distribution = {x_bar:d.prob(y) for x_bar,d in self.distributions.items()}
        seen_before = any([y in d.memory for d in self.distributions.values()])
        assert seen_before, f'learner hasn\'t seen word \'{y}\' before'
        return normalize_dict(inverse_distribution)


class LanguageAcquirer():
    def __init__(self,base_lexicon):
        self.base_lexicon = base_lexicon
        self.syntax_learner = Learner(1,CCGDirichletProcess)
        self.shell_meaning_learner = Learner(1000,ShellMeaningDirichletProcess)
        self.meaning_learner = Learner(500,MeaningDirichletProcess)
        self.word_learner = Learner(1,WordSpanDirichletProcess)
        self.lf_splits_cache = {}

    def show_word(self,word):
        distr = self.word_learner.inverse_distribution(word)
        probs = sorted(distr.items(), key=lambda x:x[1])[-15:]
        print(f'\nLearned Meaning for \'{word}\'')
        print('\n'.join([f'{prob:.3f}: {word}' for word,prob in probs]))

    def show_splits(self,syn_cat):
        return self.syntax_learner.distributions(syn_cat)

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
    for i,dpoint in enumerate(d['data'][:ndps]):
        if ARGS.simple_example:
            lf_str = 'loc colorado virginia'
            words = ['colarado', 'is', 'in', 'virginia']
        else:
            words, lf_str = dpoint['words'], dpoint['parse']
        if words[-1] == '?':
            words = words[:-1]
        print(words,lf_str)
        if i == ARGS.db_at:
            breakpoint()
        if len(words) > ARGS.max_sent_len:
            print(f"excluding because too long: {' '.join(words)}")
            continue
        language_acquirer.train_one_step(words,lf_str)
    language_acquirer.syntax_learner.distributions['S'].show()
    language_acquirer.show_word('virginia')
    language_acquirer.show_word('cities')
    print(f'Total run time: {time()-start_time:.3f}s')
    breakpoint()
