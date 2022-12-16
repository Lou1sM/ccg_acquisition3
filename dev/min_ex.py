import json
from utils import split_respecting_brackets, is_bracketed, all_sublists
import argparse
import re


class LogicalForm:
    def __init__(self,defining_string,parent=None):
        had_surrounding_brackets = False
        self.parent = parent
        if '.' in defining_string:
            lambda_string, _, remaining_string = defining_string.partition('.')
            assert bool(re.match(r'lambda \$\d',lambda_string))
            variable_index = lambda_string[-2:]
            self.is_leaf = False
            self.string = lambda_string
            self.node_type = 'lmbda'
            self.children = [LogicalForm(remaining_string)]

            for d in self.descendents():
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
                else:
                    self.node_type = 'const'
        if had_surrounding_brackets:
            assert self.subtree_as_string() == defining_string[1:-1], f'string of created node, {self.subtree_as_string()} is not equal to defining string {defining_string}'
        else:
            assert self.subtree_as_string() == defining_string, f'string of created node, {self.subtree_as_string()} is not equal to defining string {defining_string}'

    #@property
    def descendents(self):
        #return [self] + [x for item in self.children for x in item.descendents()]
        desc = [self]
        for item in self.children:
            for x in item.descendents:
                desc.append(x)
        return desc


