import json
from utils import split_respecting_brackets, is_bracketed, all_sublists, remove_possible_outer_brackets
import argparse
import re


class LogicalForm:
    def __init__(self,defining_string,parent=None):
        had_surrounding_brackets = False
        self.var_descendents = []
        self.parent = parent
        if '.' in defining_string:
            lambda_string, _, remaining_string = defining_string.partition('.')
            assert bool(re.match(r'lambda \$\d',lambda_string))
            variable_index = lambda_string[-2:]
            self.is_leaf = False
            self.string = lambda_string
            self.node_type = 'lmbda'
            self.children = [LogicalForm(remaining_string,parent=self)]

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
                self.children = [LogicalForm(a,parent=self) for a in arguments]
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
            assert self.subtree_as_string() == defining_string[1:-1], f'string of created node, {self.subtree_as_string()} is not equal to defining string {defining_string}'
        else:
            assert self.subtree_as_string() == defining_string, f'string of created node, {self.subtree_as_string()} is not equal to defining string {defining_string}'

    @property
    def right_siblings(self):
        if self.parent is None:
            return [self]
        siblings = self.parent.children
        idx = siblings.index(self)
        return siblings[idx+1:]

    @property
    def descendents_and_right_siblings(self):
        return [d for s in [self]+self.right_siblings for d in s.descendents]

    @property
    def descendents(self):
        return [self] + [x for item in self.children for x in item.descendents]

    @property
    def leaf_descendents(self):
        if self.is_leaf:
            return [self]
        return [x for item in self.children for x in item.leaf_descendents]

    @property
    def BFSdescendents(self):
        descendents = []
        frontier = [self]
        while len(frontier) > 0:
            new = frontier.pop()
            frontier += new.children
            descendents.append(new)
        return descendents

    @property
    def num_var_descendents(self):
        num = len(self.var_descendents)
        return num

    @property
    def new_var_num(self):
        return '$0' if self.var_descendents == [] else max(self.var_descendents)+1

    @property
    def max_var_index(self):
        var_occurrences = re.findall(r'\$\d',self.subtree_as_string())
        var_indices = [int(v[1]) for v in var_occurrences]
        return -1 if len(var_indices)==0 else max(var_indices)

    @property
    def num_lambda_binders(self):
        maybe_lambda_list = self.subtree_as_string().split('.')
        assert all([x.startswith('lambda') for x in maybe_lambda_list[:-1]])
        return len(maybe_lambda_list)-1

    def extend_var_descendents(self,var_num):
        if var_num not in self.var_descendents:
            self.var_descendents.append(var_num)
        if self.parent is not None:
            self.parent.extend_var_descendents(var_num)

    def subtree_as_string(self,show_treelike=False):
        x = self.subtree_as_string_(show_treelike)
        assert is_bracketed(x) or ' ' not in x or self.node_type == 'lmbda'
        if is_bracketed(x):
            return x[1:-1]
        else:
            return x

    def subtree_as_string_(self,show_treelike):
        if self.node_type == 'lmbda':
            subtree_string = self.children[0].subtree_as_string(show_treelike)
            if show_treelike:
                subtree_string = subtree_string.replace('\n\t','\n\t\t')
                return f'{self.string}\n\t{subtree_string}\n'
            else:
                return f'{self.string}.{subtree_string}'
        elif self.node_type == 'composite':
            child_trees = [c.subtree_as_string_(show_treelike) for c in self.children]
            if show_treelike:
                subtree_string = '\n\t'.join([c.replace('\n\t','\n\t\t') for c in child_trees])
                return f'{self.string}\n\t{subtree_string}'
            else:
                subtree_string = ' '.join(child_trees)
                return f'{self.string}({subtree_string})'
        else:
            return self.string

    def copy(self):
        copied_version = LogicalForm(self.subtree_as_string())
        assert copied_version == self
        return copied_version

    def __repr__(self):
        return f'LogicalForm of type {self.node_type.upper()}: {self.subtree_as_string()}'

    def __eq__(self,other):
        if not isinstance(other,LogicalForm):
            return False
        return other.subtree_as_string() == self.subtree_as_string()

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
        copied = LogicalForm(lambda_prefix+'.' + copied.subtree_as_string())
        return copied

    def all_splits(self):
        if self.is_leaf:
            return []
        if self.num_lambda_binders > 4:
            return []
        splits = []
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
            print('entry point', entry_point)
            g = entry_point.copy()
            to_present_as_args_to_g = [d for d in entry_point.leaf_descendents if d not in to_remove]
            g = g.turn_nodes_to_vars(to_present_as_args_to_g)
            if g.num_lambda_binders > 4:
                continue
            g_sub_var_num = max(self.var_descendents)+2
            #new_entry_point_in_f_as_str = ' '.join([f'${g_sub_var_num}'] + list(set([n.string for n in to_present_as_args_to_g])))
            new_entry_point_in_f_as_str = ' '.join([f'${g_sub_var_num}'] + list(reversed([n.string for n in to_present_as_args_to_g])))
            entry_point.__init__(new_entry_point_in_f_as_str)
            f.__init__(f'lambda ${g_sub_var_num}.' + f.subtree_as_string())
            if f.num_lambda_binders > 4:
                continue
            concatted = concat_lfs(f,g)
            print(beta_normalize(concatted))
            if not beta_normalize(concatted) == self.subtree_as_string():
                breakpoint()
            beta_normalize(concatted)
            splits.append((f,g))
        return splits

def concat_lfs(lf1,lf2):
    return '(' + lf1.subtree_as_string() + ') (' + lf2.subtree_as_string() + ')'

class TreeNode():
    def __init__(self,lf,words,parent,parent_rule=None):
        self.logical_form = lf
        self.words = words
        self.possible_splits = []
        self.parent = parent
        self.parent_rule = parent_rule
        if self.parent == 'START':
            self.syntactic_category = 'S'
        elif self.logical_form.string in NPS:
            self.syntactic_category = 'NP'
        elif self.logical_form.string in INTRANSITIVES:
            self.syntactic_category = 'S|NP'
        elif self.logical_form.string in TRANSITIVES:
            self.syntactic_category = 'S|NP|NP'
        else:
            self.syntactic_category = 'PLACEHOLDER'

        for f,g in self.logical_form.all_splits():
            for split_point in range(1,len(self.words)-1):
                left_words = self.words[:split_point]
                right_words = self.words[split_point:]
                left_child_forward = TreeNode(f,left_words,parent=self,parent_rule='fwd')
                right_child_forward = TreeNode(g,right_words,parent=self,parent_rule='fwd')
                self.possible_splits.append((left_child_forward,right_child_forward))
                left_child_backward = TreeNode(g,left_words,parent=self,parent_rule='bckwd')
                right_child_backward = TreeNode(f,right_words,parent=self,parent_rule='bckwd')
                self.possible_splits.append((left_child_backward,right_child_backward))

    def __repr__(self):
        return (f"Treenode\nWords: {' '.join(self.words)}\n"
                f"With {self.logical_form.__repr__()}\n"
                f"and syn-cat {self.syntactic_category}\n")

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
    # left-associative, so recurse to the left
    left_ = remove_possible_outer_brackets(' '.join(splits[:-1]))
    #left = '('+beta_normalize(left_)+')'
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
        #return '('+beta_normalize(combined)+')'
        return beta_normalize(combined)
    else:
        #if re.match(r'^[\w\$]*$',right):
            #return f'{left} {right}'
        #else:
            #return f'{left} ({right})'
        return ' '.join([left,right])


def shift_variable_names(lf_as_string,shift_from=[0]):
    for sf in shift_from:
        var_occurrences = re.findall(r'\$\d',lf_as_string)
        var_idxs_to_shift = set([int(v[1]) for v in var_occurrences if int(v[1]) >= sf])
        for vidx in sorted(var_idxs_to_shift,reverse=True):
            lf_as_string = re.sub(f'\${vidx}',f'${vidx+1}',lf_as_string)
    for sf in shift_from:
        assert f'${sf}' not in lf_as_string
    return lf_as_string


ARGS = argparse.ArgumentParser()
ARGS.add_argument("--expname", type=str, default='tmp',
                      help="the directory to write output files")
ARGS.add_argument("--dset", type=str, choices=['easy-adam','geo'], default="geo")
ARGS.add_argument("--is_dump_verb_repo", action="store_true",
                      help="whether to dump the verb repository")
ARGS.add_argument("--devel", "--development_mode", action="store_true")
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
    lf = LogicalForm(parse)
    splits = lf.all_splits()
    if '.' in words:
        breakpoint()
    tn = TreeNode(lf,words,'START')
    print(tn)
    for f,g in tn.possible_splits:
        print(f)
        print(g)
        print('\n')
    breakpoint()
    break
