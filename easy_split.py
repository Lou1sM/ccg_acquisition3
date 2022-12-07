import json
from pprint import pprint
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
            assert self.subtree_string() == defining_string[1:-1], f'string of created node, {self.subtree_string()} is not equal to defining string {defining_string}'
        else:
            assert self.subtree_string() == defining_string, f'string of created node, {self.subtree_string()} is not equal to defining string {defining_string}'

        if self.stripped_string in NPS:
            self.is_syntactic_leaf = True
            self.syntactic_category = 'NP'
        elif self.stripped_string in INTRANSITIVES:
            self.is_syntactic_leaf = True
            self.syntactic_category = 'S|NP'
        elif self.stripped_string in TRANSITIVES:
            self.is_syntactic_leaf = True
            self.syntactic_category = 'S|NP|NP'
        else:
            self.is_syntactic_leaf = False
            self.syntactic_category = 'XXX'

    @property
    def stripped_string(self):
        ss = re.sub(r'lambda \$\d{1,2}\.','',self.subtree_string())
        ss = re.sub(r'[\$\d \(\)]','',ss) # assuming '$' and '\d' not in string
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
        if self.parent is not None:
            self.parent.extend_var_descendents(var_num)

    def subtree_string(self,show_treelike=False):
        x = self.subtree_string_(show_treelike)
        assert is_bracketed(x) or ' ' not in x or self.node_type == 'lmbda'
        if is_bracketed(x):
            return x[1:-1]
        else:
            return x

    def subtree_string_(self,show_treelike):
        if self.node_type == 'lmbda':
            subtree_string = self.children[0].subtree_string(show_treelike)
            if show_treelike:
                subtree_string = subtree_string.replace('\n\t','\n\t\t')
                return f'{self.string}\n\t{subtree_string}\n'
            else:
                return f'{self.string}.{subtree_string}'
        elif self.node_type == 'composite':
            child_trees = [c.subtree_string_(show_treelike) for c in self.children]
            if show_treelike:
                subtree_string = '\n\t'.join([c.replace('\n\t','\n\t\t') for c in child_trees])
                return f'{self.string}\n\t{subtree_string}'
            else:
                subtree_string = ' '.join(child_trees)
                return f'{self.string}({subtree_string})'
        else:
            return self.string

    def copy(self):
        copied_version = LogicalForm(self.subtree_string())
        assert copied_version == self
        return copied_version

    def __repr__(self):
        return f'LogicalForm of type {self.node_type.upper()}: {self.subtree_string()}'

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
        copied = LogicalForm(lambda_prefix+'.' + copied.subtree_string())
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
            g = entry_point.copy()
            to_present_as_args_to_g = [d for d in entry_point.leaf_descendents if d not in to_remove]
            g = g.turn_nodes_to_vars(to_present_as_args_to_g)
            if g.num_lambda_binders > 4:
                continue
            g_sub_var_num = self.new_var_num+1
            new_entry_point_in_f_as_str = ' '.join([f'${g_sub_var_num}'] + list(reversed([n.string for n in to_present_as_args_to_g])))
            entry_point.__init__(new_entry_point_in_f_as_str)
            f.__init__(f'lambda ${g_sub_var_num}.' + f.subtree_string())
            if f.num_lambda_binders > 4:
                continue
            concatted = concat_lfs(f,g)
            if not beta_normalize(concatted) == self.subtree_string():
                breakpoint()
            beta_normalize(concatted)
            splits.append((f,g))
        return splits

def concat_lfs(lf1,lf2):
    return '(' + lf1.subtree_string() + ') (' + lf2.subtree_string() + ')'

class ParseNode():
    def __init__(self,lf,words,node_type,parent=None,sibling=None):
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
        self.syntactic_category = self.logical_form.syntactic_category
        self.is_syntactic_leaf = self.logical_form.is_syntactic_leaf
        if self.logical_form.stripped_string in NPS:
            self.is_leaf = True
            self.syntactic_category_placeholder = 'NP'
        elif self.logical_form.stripped_string in INTRANSITIVES:
            self.is_leaf = True
            self.syntactic_category_placeholder = 'S|NP'
        elif self.logical_form.stripped_string in TRANSITIVES:
            self.is_leaf = True
            self.syntactic_category_placeholder = 'S|NP|NP'
        else:
            self.is_leaf = False
            self.syntactic_category_placeholder = 'XXX'

            if len(self.words) == 1:
                return

            for f,g in self.logical_form.all_splits():
                if f.is_syntactic_leaf and g.is_syntactic_leaf:
                    if re.match(fr'.*\|{g.syntactic_category}',f.syntactic_category):
                        self.add_splits(f,g,'fwd')
                        self.add_splits(f,g,'bckwd')
                else:
                    self.add_splits(f,g,'fwd')
                    self.add_splits(f,g,'bckwd')

    def add_splits(self,f,g,direction):
        for split_point in range(1,len(self.words)):
            left_words = self.words[:split_point]
            right_words = self.words[split_point:]
            # CONVENTION: child with the words on the left of sentence is
            # the 'left' child, even if bck application, left child is g in
            # that case
            if direction == 'fwd':
                left_child = ParseNode(f,left_words,parent=self,node_type='left_fwd')
                right_child = ParseNode(g,right_words,parent=self,node_type='right_fwd')
                hits = re.findall(fr'[/\|]\(?{re.escape(right_child.syntactic_category)}\)?$',left_child.syntactic_category)
                for hit in hits:
                    self.syntactic_category = left_child.syntactic_category[:-len(hit)]
            elif direction == 'bckwd':
                right_child = ParseNode(f,right_words,parent=self,node_type='right_bckwd')
                left_child = ParseNode(g,left_words,parent=self,node_type='right_fwd')
                hits = re.findall(fr'[\\\|]\(?{re.escape(right_child.syntactic_category)}\)?$',left_child.syntactic_category)
            for hit in hits:
                self.syntactic_category_placeholder = left_child.syntactic_category[:-len(hit)-1]
            if  (not left_child.is_leaf and len(left_child.possible_splits) == 0 or
                 not right_child.is_leaf and len(right_child.possible_splits) == 0):
                continue
            self.append_split(left_child,right_child,direction)

    def append_split(self,left,right,combinator):
        left.siblingify(right)
        assert left.sibling is not None
        assert right.sibling is not None
        self.possible_splits.append({'left':left,'right':right,'combinator':combinator})

    @property
    def is_g(self):
        return self.node_type in ['right_fwd','left_bckwd']

    def is_fwd(self):
        return self.node_type in ['right_fwd','left_fwd']
    @property
    def syn_cat_is_set(self):
        return self.syntactic_category_placeholder.is_set

    def __repr__(self):
        return (f"ParseNode\n"
                f"\tWords: {' '.join(self.words)}\n"
                f"\tLogical Form: {self.logical_form.subtree_string()}\n"
                f"\tSyntactic Category: {self.syntactic_category}\n")

    def siblingify(self,other):
        self.sibling = other
        other.sibling = self


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

if __name__ == "__main__":
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
        #pn = ParseNode(lf,words,'ROOT')
        lf = LogicalForm('loc colorado virginia')
        pn = ParseNode(lf, ['colarado', 'is', 'in', 'virginia'],'ROOT')
        print(pn)
        for ps in pn.possible_splits:
            pprint(ps)
            print('\n')
        breakpoint()
        break
