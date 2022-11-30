import numpy as np
from copy import deepcopy
import re


with open('data/easy_training_examples.txt') as f:
    data = f.readlines()

X = [x[6:-1] for x in data if x.startswith('Sent')]
y = [x[5:-1] for x in data if x.startswith('Sem')]


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

    def prepend_new_lamba(self):
        shifted_lf_string = shift_variable_names(self.subtree_as_string())
        new_lf_string = 'lambda $0.' + shifted_lf_string
        breakpoint()
        copied_version = LogicalForm(new_lf_string)
        assert len(copied_version.list_descendents()) == len(self.list_descendents())+1
        for orig,copied, in zip(self.list_descendents(),copied_version.list_descendents()[1:]):
            assert copied.node_type == orig.node_type
        return copied_version

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

    def split_off_leaves(self):
        splits = []
        for to_remove_idx, d in enumerate(self.list_descendents()):
            if not d.is_leaf or d==self or d.node_type=='bound_var':
                continue
            f = self.prepend_new_lamba()
            node_being_removed = f.list_descendents()[to_remove_idx+1]
            assert node_being_removed == d
            node_being_removed.node_type = 'bound_var'
            node_being_removed.string = '$0'
            g = d.copy()
            splits.append((f,g))
        return splits


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

def shift_variable_names(lf_as_string):
    var_occurrences = re.findall(r'\$\d',lf_as_string)
    var_idxs_used = set([int(v[1]) for v in var_occurrences])
    for vidx in sorted(var_idxs_used,reverse=True):
        lf_as_string = re.sub(f'\${vidx}',f'${vidx+1}',lf_as_string)
    return lf_as_string


for inp_string in y:
    inp_string = re.sub(r'_\d','',inp_string)
    inp_string = re.sub(r'_\{[er]\}','',inp_string)
    inp_string = re.sub(r'[\w:]+\|','',inp_string)
    lf = LogicalForm(inp_string)
    splits = lf.split_off_leaves()
    for f,g in splits:
        print(f.subtree_as_string(), g.subtree_as_string())
    break
