import json
from utils import translate, translate_by_unify
from pprint import pprint
from utils import split_respecting_brackets, is_bracketed, all_sublists, remove_possible_outer_brackets, maybe_bracketted
import argparse
import re


# is_leaf means its atomic in lambda calculus
# is_semantic_leaf means we shouldn't consider breaking it further
# e.g. lambda $0. state $0 is a semantic leaf but a leaf
class LogicalForm:
    def __init__(self,defining_string,base_lexicon,splits_cache,shell_splits_cache,parent=None):
        had_surrounding_brackets = False
        self.base_lexicon = base_lexicon
        self.splits_cache = splits_cache
        self.shell_splits_cache = shell_splits_cache
        self.var_descendents = []
        self.possible_splits = []
        self.parent = parent
        self.stored_subtree_string = ''
        self.stored_shell_subtree_string = ''
        self.stored_alpha_normalized_subtree_string = ''
        self.stored_alpha_normalized_shell_subtree_string = ''
        if parent == 'START':
            self.sem_cat = 'S'
            self.is_semantic_leaf = False

        if '.' in defining_string:
            lambda_string, _, remaining_string = defining_string.partition('.')
            assert bool(re.match(r'lambda \$\d',lambda_string))
            variable_index = lambda_string[-2:]
            self.is_leaf = False
            self.string = lambda_string
            self.node_type = 'lmbda'
            self.children = [self.spawn_child(remaining_string)]

            for d in self.descendents:
                if d.node_type == 'unbound_var' and d.string == variable_index:
                    d.binder = self
                    d.node_type = 'bound_var'
        else:
            if ' ' in defining_string:
                self.string = ''
                self.node_type = 'composite'
                arguments = split_respecting_brackets(defining_string)
                if len(arguments) == 1 and is_bracketed(defining_string):
                    had_surrounding_brackets = True
                    assert not is_bracketed(defining_string[1:-1])
                    arguments = split_respecting_brackets(defining_string[1:-1])
                self.children = [self.spawn_child(a) for a in arguments]
                self.is_leaf = False
            else:
                self.string = defining_string
                self.children = []
                self.is_leaf = True
                if self.string.startswith('$'):
                    self.node_type = 'unbound_var'
                    self.extend_var_descendents(int(self.string[1:]))
                elif self.string == 'AND':
                    self.node_type = 'connective'
                else:
                    self.node_type = 'const'
        if had_surrounding_brackets:
            assert self.subtree_string() == defining_string[1:-1]
        else:
            assert self.subtree_string() == defining_string

        if self.parent != 'START':
            ss = strip_string(defining_string)
            if ss in self.base_lexicon:
                self.sem_cat = self.base_lexicon[ss]
                self.is_semantic_leaf = True
            else:
                self.sem_cat = 'XXX'
                self.is_semantic_leaf = False
        if defining_string == "lambda $2.lambda $1.loc $1 $2":
            assert self.sem_cat == 'S|NP|NP'

    def spawn_child(self,defining_string):
        return LogicalForm(defining_string,base_lexicon=self.base_lexicon,splits_cache=self.splits_cache,shell_splits_cache=self.shell_splits_cache,parent=self)

    def spawn_self_like(self,defining_string):
        new = LogicalForm(defining_string,base_lexicon=self.base_lexicon,splits_cache=self.splits_cache,shell_splits_cache=self.shell_splits_cache,parent=self.parent)
        new.sem_cat = self.sem_cat
        return new

    @property
    def stripped_subtree_string(self):
        ss = re.sub(r'lambda \$\d{1,2}\.','',self.subtree_string())
        ss = re.sub(r'( \$\d)+$','',ss).replace('()','')
        # allowed to have trailing bound variables
        return strip_string(self.subtree_string())
        return ss

    @property
    def descendents(self):
        return [self] + [x for item in self.children for x in item.descendents]

    @property
    def leaf_descendents(self):
        if self.is_leaf:
            return [self]
        return [x for item in self.children for x in item.leaf_descendents]

    @property
    def new_var_num(self):
        return 0 if self.var_descendents == [] else max(self.var_descendents)+1

    @property
    def num_lambda_binders(self):
        maybe_lambda_list = self.subtree_string().split('.')
        assert all([x.startswith('lambda') for x in maybe_lambda_list[:-1]])
        return len(maybe_lambda_list)-1

    def extend_var_descendents(self,var_num):
        if var_num not in self.var_descendents:
            self.var_descendents.append(var_num)
        if self.parent is not None and self.parent != 'START':
            self.parent.extend_var_descendents(var_num)

    def set_subtree_string(self,as_shell,alpha_normalized,string):
        if as_shell and alpha_normalized:
            self.stored_alpha_normalized_shell_subtree_string = string
        elif as_shell and not alpha_normalized:
            self.stored_shell_subtree_string = string
        elif not as_shell and alpha_normalized:
            assert 'PLACEHOLDER' not in string
            self.stored_alpha_normalized_subtree_string = string
        else:
            assert 'PLACEHOLDER' not in string
            self.stored_subtree_string = string

    def get_stored_subtree_string(self,as_shell,alpha_normalized):
        if as_shell and alpha_normalized:
            return self.stored_alpha_normalized_shell_subtree_string
        elif as_shell and not alpha_normalized:
            return self.stored_shell_subtree_string
        elif not as_shell and alpha_normalized:
            return self.stored_alpha_normalized_subtree_string
        else:
            return self.stored_subtree_string

    def subtree_string(self,alpha_normalized=False,as_shell=False,recompute=False,show_treelike=False):
        x = self.get_stored_subtree_string(as_shell,alpha_normalized)
        if x == '' or recompute:
            x = self.subtree_string_(show_treelike=show_treelike,as_shell=as_shell,recompute=recompute)
            assert is_bracketed(x) or ' ' not in x or self.node_type == 'lmbda'
            if is_bracketed(x):
                x = x[1:-1]
            if alpha_normalized:
                trans_list = sorted(self.var_descendents)
                for v_new, v_old in enumerate(trans_list):
                    x = re.sub(fr'\${v_old}',f'${v_new}',x)
        return x

    def subtree_string_(self,show_treelike,as_shell,recompute):
        if self.node_type == 'lmbda':
            subtree_string = self.children[0].subtree_string(as_shell=as_shell,show_treelike=show_treelike,recompute=recompute)
            if show_treelike:
                subtree_string = subtree_string.replace('\n\t','\n\t\t')
                x = f'{self.string}\n\t{subtree_string}\n'
            else:
                x = f'{self.string}.{subtree_string}'
        elif self.node_type == 'composite':
            if as_shell:
                child_trees = ['PLACEHOLDER' if c.node_type == 'const' else c.subtree_string_(show_treelike,as_shell,recompute=recompute) for c in self.children]
            else:
                child_trees = [c.subtree_string_(show_treelike,as_shell,recompute=recompute) for c in self.children]
            if show_treelike:
                subtree_string = '\n\t'.join([c.replace('\n\t','\n\t\t') for c in child_trees])
                x = f'{self.string}\n\t{subtree_string}'
            else:
                subtree_string = ' '.join(child_trees)
                x = f'{self.string}({subtree_string})'
        else:
            x = self.string

        assert as_shell or 'PLACEHOLDER' not in x
        if recompute:
            self.set_subtree_string(as_shell,alpha_normalized=False,string=x)

        return x

    def copy(self):
        copied_version = self.spawn_self_like(self.subtree_string())
        assert copied_version == self
        copied_version.sem_cat = self.sem_cat
        return copied_version

    def __repr__(self):
        return (f'LogicalForm of type {self.node_type.upper()}: {self.subtree_string()}'
                f'\n\tsemantic category: {self.sem_cat}')

    def __hash__(self):
        return hash(self.subtree_string(alpha_normalized=True))

    def __eq__(self,other):
        if not isinstance(other,LogicalForm):
            return False
        return other.subtree_string() == self.subtree_string()

    def turn_nodes_to_vars(self,nodes):
        copied = self.copy()
        if len(nodes) == 0:
            return copied
        to_remove = [d for d in copied.descendents if d in nodes]
        # node may be 'in' another list if there is another node in that list
        # that has the same defining string as node, but we want the reference,
        # i.e. node itself, rather than its copy in the list
        vars_abstracted = []
        for desc in to_remove:
            var_num = copied.new_var_num
            desc.string = f'${var_num}'
            vars_abstracted.append(var_num)
            desc.extend_var_descendents(var_num)
            desc.node_type = 'bound_var'
        lambda_prefix = '.'.join([f'lambda ${vn}' for vn in reversed(vars_abstracted)])
        copied = self.spawn_self_like(lambda_prefix+'.' + copied.subtree_string())
        return copied

    def lambda_abstract(self,var_num):
        new = self.spawn_self_like(f'lambda ${var_num}')
        new.node_type = 'lmbda'
        new.children = [self]
        new.stored_subtree_string =f'lambda ${var_num}.{self.subtree_string()}'
        new.stored_alpha_normalized_subtree_string =f'lambda ${var_num}.{self.subtree_string(alpha_normalized=True)}'
        new.stored_shell_subtree_string =f'lambda ${var_num}.{self.subtree_string(as_shell=True)}'
        new.stored_alpha_normalized_shell_subtree_string =f'lambda ${var_num}.{self.subtree_string(alpha_normalized=True,as_shell=True)}'
        return new

    def infer_splits(self):
        if self.is_semantic_leaf:
            return []
        if self.num_lambda_binders > 4:
            return []
        if self in self.splits_cache:
            return self.splits_cache[self]
        #if self.subtree_string(as_shell=True,alpha_normalized=True) in self.shell_splits_cache:
        #    existing_split_head,existing_splits = self.shell_splits_cache[self.subtree_string(alpha_normalized=True,as_shell=True)]
        #    if existing_splits == []:
        #        self.possible_splits = []
        #    else:
        #        translation = translate_by_unify(existing_split_head,self.subtree_string())
        #        translate(next(iter(existing_splits))[0].subtree_string(),translation)
        #        self.possible_splits = [(f.spawn_self_like(translate(f.subtree_string(),translation)),g.spawn_self_like(translate(g.subtree_string(),translation))) for f,g in existing_splits]
        #    self.splits_cache[self] = self.possible_splits
        #    return self.possible_splits
        possible_removee_idxs = [i for i,d in enumerate(self.descendents) if d.node_type=='const']
        for removee_idxs in all_sublists(possible_removee_idxs):
            if removee_idxs == []: continue
            f = self.copy()
            to_remove = [f.descendents[i] for i in removee_idxs]
            leftmost = f.descendents[min(removee_idxs)]
            entry_point = leftmost # where to substitute g into f
            changed = True
            while changed:
                if entry_point.parent is None:
                    break
                changed = False
                for tr in to_remove:
                    if tr not in entry_point.descendents:
                        changed = True
                        entry_point = entry_point.parent
                        break
            g = entry_point.copy()
            to_present_as_args_to_g = [d for d in entry_point.leaf_descendents if d not in to_remove]
            g = g.turn_nodes_to_vars(to_present_as_args_to_g)
            if (g.num_lambda_binders > 4) or (strip_string(g.subtree_string().replace(' AND','')) == ''): # don't consider only variables
                continue
            g_sub_var_num = self.new_var_num
            new_entry_point_in_f_as_str = ' '.join([f'${g_sub_var_num}'] + list(reversed([n.string for n in to_present_as_args_to_g])))
            assert entry_point.subtree_string() in self.subtree_string()
            entry_point.__init__(new_entry_point_in_f_as_str,entry_point.base_lexicon,entry_point.splits_cache,entry_point.shell_splits_cache)
            if len(to_present_as_args_to_g) > 3: # then f will end up with arity 4
                continue
            if strip_string(f.subtree_string().replace(' AND','')) == '': # also exclude case where only substantive is 'AND'
                continue
            f = f.lambda_abstract(g_sub_var_num)
            f.sem_cat = f.base_lexicon.get(f.stripped_subtree_string,'XXX')
            concatted = concat_lfs(f,g)
            assert beta_normalize(concatted) == self.subtree_string()
            assert f != self
            assert g != self
            self.possible_splits.append((f,g))
            if g.sem_cat_is_set:
                if f.sem_cat_is_set:
                    hits = re.findall(fr'\(?{re.escape(g.sem_cat)}\)?$',f.sem_cat)
                    for hit in hits:
                        self.sem_cat = f.sem_cat[:-len(hit)-1]
                if self.sem_cat_is_set:
                    self.propagate_sem_cat_leftward(f,g)
            assert f.sem_cat_is_set or not self.sem_cat_is_set or not g.sem_cat_is_set
            f.infer_splits()
            g.infer_splits()
        self.splits_cache[self] = self.possible_splits
        x = (self.subtree_string(),self.possible_splits)
        self.shell_splits_cache[self.subtree_string(as_shell=True,alpha_normalized=True)] = x
        return self.possible_splits

    def propagate_sem_cat_leftward(self,f,g):
        if '|' in g.sem_cat:
            new_assigned_category = f'{self.sem_cat}|({g.sem_cat})'
        else:
            new_assigned_category = f'{self.sem_cat}|{g.sem_cat}'
        assert f.sem_cat in [new_assigned_category,'XXX']
        f.sem_cat = new_assigned_category
        for f1,g1 in f.possible_splits:
            if g1.sem_cat_is_set:
                f.propagate_sem_cat_leftward(f1,g1)

    @property
    def sem_cat_is_set(self):
        return self.sem_cat != 'XXX'

def concat_lfs(lf1,lf2):
    return '(' + lf1.subtree_string() + ') (' + lf2.subtree_string() + ')'

class ParseNode():
    def __init__(self,lf,words,node_type,parent=None,sibling=None,syn_cat=None):
        self.logical_form = lf
        self.words = words
        self.possible_splits = []
        self.parent = parent
        self.node_type = node_type
        self.sibling = sibling
        assert (parent is None) or (node_type != 'ROOT'), \
            f"you're trying to specify a parent for the root node"
        assert (parent is not None) or (node_type == 'ROOT'), \
            f"you're failing to specify a parent for a non-root node"
        assert (sibling is None) or (node_type != 'ROOT'), \
            f"you're trying to specify a sibling for the root node"
        self.sem_cat = self.logical_form.sem_cat
        if syn_cat is not None:
            self.syn_cat = syn_cat
        elif self.node_type == 'ROOT':
            self.syn_cat = 'S'
            assert self.sem_cat == 'S'
        else:
            self.syn_cat = self.logical_form.sem_cat # no directionality then
        self.is_leaf = self.logical_form.is_semantic_leaf or len(self.words) == 1
        if not self.is_leaf:
            if self.logical_form.possible_splits == []:
                self.logical_form.infer_splits()
            for f,g in self.logical_form.possible_splits:
                if g.sem_cat_is_set:
                    #assert re.match(fr'.*\|\(?{g.sem_cat}\)?$',f.sem_cat)
                    self.add_splits(f,g,'fwd')
                    self.add_splits(f,g,'bck')
        #else:
            #print(self.syn_cat)

    def add_splits(self,f,g,direction):
        for split_point in range(1,len(self.words)):
            left_words = self.words[:split_point]
            right_words = self.words[split_point:]
            # CONVENTION: child with the words on the left of sentence is
            # the 'left' child, even if bck application, left child is g in
            # that case
            if direction == 'fwd':
                right_child = ParseNode(g,right_words,parent=self,node_type='right_fwd')
                new_syn_cat = f'{self.syn_cat}/{maybe_bracketted(right_child.syn_cat)}'
                assert f.sem_cat == 'XXX' or is_congruent(new_syn_cat,f.sem_cat)
                left_child = ParseNode(f,left_words,parent=self,node_type='left_fwd',syn_cat=new_syn_cat)
            elif direction == 'bck':
                left_child = ParseNode(g,left_words,parent=self,node_type='left_bck')
                new_syn_cat = f'{self.syn_cat}\\{maybe_bracketted(left_child.syn_cat)}'
                assert f.sem_cat == 'XXX' or is_congruent(new_syn_cat,f.sem_cat)
                right_child = ParseNode(f,right_words,parent=self,node_type='right_bck',syn_cat=new_syn_cat)
            self.append_split(left_child,right_child,direction)

        if len(self.possible_splits) == 0:
            self.is_leaf = True

    def append_split(self,left,right,combinator):
        left.siblingify(right)
        assert left.sibling is not None
        assert right.sibling is not None
        self.possible_splits.append({'left':left,'right':right,'combinator':combinator})

    def propagate_syn_cat(self,f,g):
        if '|' in g.sem_cat:
            new_assigned_category = f'{self.sem_cat}|({g.sem_cat})'
        else:
            new_assigned_category = f'{self.sem_cat}|{g.sem_cat}'
        assert f.sem_cat in [new_assigned_category,'XXX']
        f.sem_cat = new_assigned_category
        for f1,g1 in f.possible_splits:
            if g1.sem_cat_is_set:
                f.propagate_sem_cat_leftward(f1,g1)

    @property
    def is_g(self):
        return self.node_type in ['right_fwd','left_bck']

    def is_fwd(self):
        return self.node_type in ['right_fwd','left_fwd']

    def __repr__(self):
        base = (f"ParseNode\n"
                f"\tWords: {' '.join(self.words)}\n"
                f"\tLogical Form: {self.logical_form.subtree_string()}\n"
                f"\tSyntactic Category: {self.syn_cat}\n")
        if hasattr(self,'stored_prob'):
            base += f'\tProb: {self.stored_prob}\n'
        return base

    def siblingify(self,other):
        self.sibling = other
        other.sibling = self

    def prob(self,syntax_learner,shell_meaning_learner,meaning_learner,word_learner,cache):
        if self in cache:
            return cache[self]
        shell_lf = self.logical_form.subtree_string(as_shell=True,alpha_normalized=True)
        lf = self.logical_form.subtree_string(alpha_normalized=True)
        word_str = ' '.join(self.words)
        prob_as_leaf = shell_meaning_learner.prob(self.logical_form,self.sem_cat) * meaning_learner.prob(lf,shell_lf) * word_learner.prob(word_str,lf) # This is where I'm omitting the conditionality that Omri used
        prob_from_descendents = 0
        for ps in self.possible_splits:
            syntax_split = ps['left'].syn_cat + ' + ' + ps['right'].syn_cat
            split_prob = syntax_learner.prob(syntax_split,self.syn_cat)
            prob_from_descendents += ps['right'].prob(syntax_learner,shell_meaning_learner,meaning_learner,word_learner,cache)*ps['left'].prob(syntax_learner,shell_meaning_learner,meaning_learner,word_learner,cache)*split_prob

        total_prob = prob_as_leaf + prob_from_descendents
        cache[self] = total_prob
        self.stored_prob = total_prob
        return total_prob

def beta_normalize(m):
    m = remove_possible_outer_brackets(m)
    if m.startswith('lambda'):
        lambda_binder = re.match(r'lambda \$\d+\.',m).group(0)
        body = m[len(lambda_binder):]
        return lambda_binder + beta_normalize(body)
    if re.match(r'^[\w\$]*$',m):
        return m
    splits = split_respecting_brackets(m)
    if len(splits) == 1:
        return m
    left_ = remove_possible_outer_brackets(' '.join(splits[:-1]))
    left = beta_normalize(left_)
    assert 'lambda (' not in left
    right_ = splits[-1]
    right = beta_normalize(right_)
    if not re.match(r'^[\w\$]*$',right):
        right = '('+right+')'

    if left.startswith('lambda'):
        lambda_binder,_,rest = left.partition('.')
        assert re.match(r'lambda \$\d{1,2}',lambda_binder)
        var_name = lambda_binder[7:]
        assert re.match(r'\$\d',var_name)
        combined = re.sub(re.escape(var_name),right,rest)
        return beta_normalize(combined)
    else:
        return ' '.join([left,right])

def strip_string(ss):
    ss = re.sub(r'lambda \$\d+\.','',ss)
    ss = re.sub(r'( ?\$\d+)+$','',ss.replace('(','').replace(')',''))
    # allowed to have trailing bound variables
    return ss

def is_congruent(syn_cat,sem_cat):
    assert '\\' not in sem_cat and '/' not in sem_cat
    sem_cat_splits = split_respecting_brackets(sem_cat,sep='|')
    syn_cat_splits = split_respecting_brackets(syn_cat,sep=['\\','/','|'])
    return sem_cat_splits == syn_cat_splits
if __name__ == "__main__":
    ARGS = argparse.ArgumentParser()
    ARGS.add_argument("--expname", type=str, default='tmp',
                          help="the directory to write output files")
    ARGS.add_argument("--dset", type=str, choices=['easy-adam','geo'], default="geo")
    ARGS.add_argument("--is_dump_verb_repo", action="store_true",
                          help="whether to dump the verb repository")
    ARGS.add_argument("--devel", "--development_mode", action="store_true")
    ARGS.add_argument("--show_splits", action="store_true")
    ARGS.add_argument("--simple_example", action="store_true")
    ARGS = ARGS.parse_args()

    if ARGS.dset == 'easy-adam':
        with open('data/easy_training_examples.txt') as f:
            data = f.readlines()

        X = [re.sub(r'_\d|_\{[er]\}|[\w:]+\|','',x[6:-1]) for x in data if x.startswith('Sent')]
        y = [x[5:-2] for x in data if x.startswith('Sem')]

    else:
        with open('data/preprocessed_geoqueries.json') as f: d=json.load(f)

    NPS = d['np_list']
    TRANSITIVES = d['transitive_verbs']
    INTRANSITIVES = d['intransitive_verbs']

    for dpoint in d['data']:
        words, parse = dpoint['words'], dpoint['parse']
        if ARGS.simple_example:
            lf = LogicalForm('loc colorado virginia')
            pn = ParseNode(lf, ['colarado', 'is', 'in', 'virginia'],syn_cat='S',node_type='ROOT')
        else:
            lf = LogicalForm(parse,'START')
            pn = ParseNode(lf,words,'ROOT')
        print(pn)
        if ARGS.show_splits:
            for ps in pn.possible_splits:
                pprint(ps)
                print('\n')
        breakpoint()
        break
