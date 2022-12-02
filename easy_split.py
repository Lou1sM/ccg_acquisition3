import json
import argparse
import re


class LogicalForm:
    def __init__(self,defining_string):
        if '.' in defining_string:
            lambda_string, _, remaining_string = defining_string.partition('.')
            assert bool(re.match(r'lambda \$\d',lambda_string))
            variable_index = lambda_string[-2:]
            self.is_leaf = False
            self.string = lambda_string
            self.node_type = 'lmbda'
            self.children = [LogicalForm(remaining_string)]

            for d in self.list_descendents():
                if d.node_type == 'unbound_var' and d.string == variable_index:
                    d.binder = self
                    d.node_type = 'bound_var'
        else:
            head_string, _, possible_args = defining_string.partition('(')
            self.string = head_string
            if len(possible_args) > 0:
                self.node_type = 'pred'
                arguments = split_respecting_parentheses(possible_args[:-1]) # -1 removes final ')'
                self.children = [LogicalForm(a) for a in arguments]
                self.is_leaf = False
            else:
                self.children = []
                self.is_leaf = True
                if self.string.startswith('$'):
                    self.node_type = 'unbound_var'
                else:
                    self.node_type = 'const'
        assert self.subtree_as_string() == defining_string

    def list_descendents(self):
        return [self] + [x for item in self.children for x in item.list_descendents()]

    def subtree_as_string(self,show_treelike=False):
        if self.node_type == 'lmbda':
            subtree_string = self.children[0].subtree_as_string(show_treelike)
            if show_treelike:
                subtree_string = subtree_string.replace('\n\t','\n\t\t')
                return f'{self.string}\n\t{subtree_string}\n'
            else:
                return f'{self.string}.{subtree_string}'
        elif self.node_type == 'pred':
            child_trees = [c.subtree_as_string(show_treelike) for c in self.children]
            if show_treelike:
                subtree_string = '\n\t'.join([c.replace('\n\t','\n\t\t') for c in child_trees])
                return f'{self.string}\n\t{subtree_string}'
            else:
                subtree_string = ','.join(child_trees)
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

    @property
    def max_var_index(self):
        var_occurrences = re.findall(r'\$\d',self.subtree_as_string())
        var_indices = [int(v[1]) for v in var_occurrences]
        return -1 if len(var_indices)==0 else max(var_indices)

    def split_off_leaves(self):
        splits = []
        new_lf_string = f'lambda ${self.max_var_index+1}.' + self.subtree_as_string()
        for to_remove_idx, d in enumerate(self.list_descendents()):
            if not d.is_leaf or d==self or d.node_type=='bound_var':
                continue
            f = LogicalForm(new_lf_string)
            assert len(f.list_descendents()) == len(self.list_descendents())+1
            for orig,copied, in zip(self.list_descendents(),f.list_descendents()[1:]):
                assert copied.node_type == orig.node_type
            node_being_removed = f.list_descendents()[to_remove_idx+1]
            assert node_being_removed == d
            node_being_removed.node_type = 'bound_var'
            node_being_removed.string = f'${self.max_var_index+1}'
            g = d.copy()
            splits.append((f,g))
        return splits


class TreeNode():
    def __init__(self,lf,words,syntactic_category='TODO'):
        self.logical_form = lf
        self.words = words
        self.syntactic_category = syntactic_category
        self.possible_splits = []
        lf_splits = self.logical_form.split_off_leaves()
        for f,g in lf_splits:
            for split_point in range(1,len(self.words)-1):
                left_words = self.words[:split_point]
                right_words = self.words[split_point:]
                left_child_forward = TreeNode(f,left_words)
                right_child_forward = TreeNode(g,right_words)
                self.possible_splits.append((left_child_forward,right_child_forward))
                left_child_backward = TreeNode(g,left_words)
                right_child_backward = TreeNode(f,right_words)
                self.possible_splits.append((left_child_backward,right_child_backward))

    def __repr__(self):
        return (f"Treenode\nWords: {' '.join(self.words)}\n"
                f"With {self.logical_form.__repr__()}\n")


def split_respecting_parentheses(s):
    """Only make a split when there are no open parentheses."""
    num_open_brackets = 0
    split_points = [-1]
    for i,c in enumerate(s):
        if c == ',' and num_open_brackets == 0:
            split_points.append(i)
        elif c == '(':
            num_open_brackets += 1
        elif c == ')':
            num_open_brackets -= 1
    split_points.append(len(s))
    splits = [s[split_points[i]+1:split_points[i+1]] for i in range(len(split_points)-1)]
    return splits

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
    with open('preprocessed_geoqueries.json') as f: d=json.load(f)

for dpoint in d['data']:
    words, parse = dpoint['words'], dpoint['parse']
    lf = LogicalForm(parse)
    splits = lf.split_off_leaves()
    tn = TreeNode(lf,words)
    print(tn)
    for f,g in tn.possible_splits:
        print(f)
        print(g)
        print('\n')
    breakpoint()
    break
