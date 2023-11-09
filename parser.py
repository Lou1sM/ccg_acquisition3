import numpy as np
from utils import split_respecting_brackets, is_bracketed, all_sublists, maybe_brac, is_atomic, strip_string, cat_components, is_congruent, alpha_normalize, maybe_debrac, f_cmp_from_parent_and_g, combine_lfs, logical_type_raise, maybe_de_type_raise, logical_de_type_raise, is_wellformed_lf, is_type_raised, new_var_num, n_lambda_binders, set_congruent, lf_sem_congruent, is_cat_type_raised, lambda_match
import re
import sys; sys.setrecursionlimit(500)
from config import pos_marking_dict


# is_leaf means it's atomic in lambda calculus
# is_semantic_leaf means we shouldn't consider breaking it further
# e.g. lambda $0. runs $0 is a semantic leaf but not a leaf

# split_prob is the prob, according to the syntax_learner, of the split that
# gave birth to it; above_prob is the prob of the branch this node is on, equal
# to split_prob*parent.above_prob*sibling.below_prob; below_prob is the prob of
# this node, under all possible ways of splitting it down to the leaves;

# Q(.) is like a composite, or composite is like a dummy wrapper around an lf,
# Q then being an alternative non-dummy wrapper

BASE_LEXICON = {k:set(['NP']) for k in ('you','i','me','he','she','it')}

class LogicalForm:
    #def __init__(self,defining_string,idx_in_tree=[],cache=None,parent=None):
    def __init__(self,defining_string,idx_in_tree=[],cache={},parent=None):
        assert isinstance(idx_in_tree,list)
        assert is_wellformed_lf(defining_string)
        had_surrounding_brackets = False
        self.sibling = None
        self.idx_in_tree = idx_in_tree
        #self.cache = {'splits':{},'sem_cats':{},'lfs':{}} if cache is None else cache
        self.cache = cache
        self.var_descendents = []
        self.possible_app_splits = []
        self.possible_cmp_splits = []
        self.parent = parent #None if logical root, tmp if will be reassigned, e.g. in infer_splits
        self.stored_subtree_string = ''
        self.stored_shell_subtree_string = ''
        self.stored_alpha_normalized_subtree_string = ''
        self.stored_alpha_normalized_shell_subtree_string = ''
        self.ud_pos = ''
        defining_string = maybe_debrac(defining_string)
        if parent == 'START':
            if defining_string.startswith('Q'):
                self.sem_cats = set(['Sq'])
            elif lambda_match(defining_string):
                self.sem_cats = set(['Swhq'])
            else:
                self.sem_cats = set(['S'])

        #if re.match(r'BARE\([a-z-A-Z0-9\-_:]*\)',defining_string):
        if defining_string.startswith('BARE'):
            self.node_type = 'barenoun'
            self.is_leaf = True
            self.string = defining_string
            for v_num in set(re.findall(r'(?<=\$)\d{1,2}',defining_string)):
                self.extend_var_descendents(v_num)
        elif defining_string=='not':
            self.node_type = 'neg'
            self.string = 'not'
            self.is_leaf = True
        #elif bool(re.match(r'^lambda \$\d{1,2}(_\{(e|r|<r,t>)\})?\.',defining_string)):
        elif bool(lambda_match(defining_string)):
            lambda_string, _, remaining_string = defining_string.partition('.')
            variable_index = lambda_string.partition('_')[0][7:]
            self.is_leaf = False
            self.string = lambda_string
            self.node_type = 'lmbda'
            self.children = [self.spawn_child(remaining_string,0)]

            for d in self.descendents:
                if d.node_type == 'unbound_var' and d.string == variable_index:
                    d.binder = self
                    d.node_type = 'bound_var'
        elif bool(re.match(r'^Q \(.*\)$',defining_string)):
            self.node_type = 'Q'
            self.string = 'Q'
            self.children = [self.spawn_child(defining_string[3:-1],0)]
            self.is_leaf = False
        elif ' ' in defining_string:
            self.string = ''
            arguments = split_respecting_brackets(defining_string)
            assert arguments != [self] # would enter a loop if so
            if len(arguments) == 1 and is_bracketed(defining_string):
                had_surrounding_brackets = True
                assert not is_bracketed(defining_string[1:-1])
                arguments = split_respecting_brackets(defining_string[1:-1])
            self.children = [self.spawn_child(a,i) for i,a in enumerate(arguments)]
            if [c.node_type for c in self.children] == ['quant','noun']:
                self.node_type = 'detnoun' # determiner + noun treated kinda like a const
            else:
                self.node_type = 'composite'
            self.is_leaf = False
        else:
            self.string = defining_string
            self.is_leaf = True
            if self.string.startswith('$'):
                self.node_type = 'unbound_var'
                self.extend_var_descendents(int(self.string[1:]))
            elif self.string == 'and':
                self.node_type = 'connective'
                self.string = 'and'
            elif pos_marking_dict.get(defining_string,None) == 'N':
                breakpoint()
                self.node_type = 'noun'
            else:
                if '|' not in defining_string and not defining_string.startswith('$') and defining_string not in ['', 'you']:
                    print(defining_string)
                    breakpoint()
                self.node_type = 'const'

        if self.is_leaf: # should only have children defined if comes from the __init__ in splits
            #assert not hasattr(self,'children') or self.children==[]
            self.children = []

        if had_surrounding_brackets:
            #assert self.subtree_string()==re.match(r'(.*\|)?(.*)',defining_string).group(2)[1:-1]
            assert self.subtree_string() == defining_string
        else:
            if not self.subtree_string() == defining_string:
                breakpoint()

        self.set_cats_from_string()
        #if defining_string.startswith('lambda $1_{r}.'):
            #breakpoint()
        pass

    def set_cats_from_string(self):
        if self.parent == 'START':
            self.is_semantic_leaf = False
        else:
            ss = self.stripped_subtree_string
            self.is_semantic_leaf = self.node_type=='barenoun' or ' ' not in ss
            if not self.is_semantic_leaf:
                self.sem_cats = set(['X'])
            elif self.node_type in ['barenoun','detnoun']:
                self.sem_cats = set(['NP'])
            elif '|' in ss:
                pos_marking = ss.split('|')[0]
                self.sem_cats = pos_marking_dict.get(pos_marking,set(['X']))
            else:
                word_level_form = ss.split('_')[0] if '_' in ss else ss
                self.sem_cats = BASE_LEXICON.get(word_level_form,set(['X']))
                #if self.sem_cats != set(['X']):
                    #self.is_semantic_leaf = True # ...but not necessary
        assert self.sem_cats!=''
        assert self.sem_cats is not None
        assert not any(sc is None for sc in self.sem_cats)
        assert isinstance(self.sem_cats,set)
        if self.string == 'pro:per|you_2':
            assert self.sem_cats == set(['NP'])
        return self.is_type_congruent()

    def is_type_congruent(self):
        lf_str = logical_de_type_raise(self.lf_str) if self.is_type_raised() else self.lf_str
        self.sem_cats = set(ssc for ssc in self.sem_cats if ssc == 'X' or lf_sem_congruent(lf_str,maybe_de_type_raise(ssc)))
        return self.sem_cats != set([])

    def is_type_raised(self):
        #return is_type_raised(self.subtree_string(as_shell=True,alpha_normalized=True))
        is_tr = any(is_cat_type_raised(ssc) for ssc in self.sem_cats)
        assert is_tr == any(is_cat_type_raised(ssc) for ssc in self.sem_cats)
        return is_tr

    def infer_splits(self):
        if self.lf_str == 'lambda $0.v|see_3 pro:per|you_2 $0':
            breakpoint()
        if self.is_semantic_leaf or self.n_lambda_binders > 4:
            self.possible_app_splits = []
            self.possible_cmp_splits = []
            return self.possible_app_splits
        #if self.lf_str in self.cache['splits']:
        if self.lf_str in self.cache:
            #self.possible_app_splits = self.cache['splits'][self.lf_str]
            self.possible_app_splits, self.possible_cmp_splits, self.sem_cats = self.cache[self.lf_str]
            return self.possible_app_splits
        possible_removee_idxs = [i for i,d in enumerate(self.descendents) if d.node_type in ['const','noun','quant','barenoun','neg']]
        for removee_idxs in all_sublists(possible_removee_idxs):
            n_removees = len(removee_idxs)
            if n_removees == 0: continue
            if self.lf_str == 'Q (v|see_3 pro:per|you_2 pro:per|it_4)' and removee_idxs == [2,3,4]:
                breakpoint()
            if n_removees == len(possible_removee_idxs):
                if self.node_type == 'Q': # in that case, only split if self is a Q node
                    f = LogicalForm('lambda $0.Q ($0)')
                    g = self.descendents[1].copy()
                    self.add_split(f,g,'app')
                continue
            if self.sem_cats.intersection({'S','S|NP'}) and \
                all(self.descendents[ri].sem_cats.intersection({'N'}) for ri in removee_idxs):
                    #print('skipping', removee_idxs, 'because noun')
                    continue # this would result in illegitimate cats, like S|N or S|NP|N
            not_removees = [d for i,d in enumerate(self.descendents) if i not in removee_idxs]
            if any(self.descendents[i].lf_str==d.lf_str for i in removee_idxs for d in not_removees):
                #print('skipping', removee_idxs, 'because all-or-nothing')
                continue # all or nothing rule
            f = self.copy()
            f.parent = self
            to_remove = [f.descendents[i] for i in removee_idxs]
            #if any([x.node_type in ('const','detnoun') and x in not_removees for x in to_remove]):
            #if any([x in not_removees and self.descendents.count(x)==1 for x in to_remove]):
            if any(x in not_removees for x in to_remove):
                breakpoint()
                continue
            if any([x.node_type=='detnoun' for x in to_remove]):
                breakpoint()
            assert all([n.node_type in ['const','barenoun','detnoun','noun','quant','neg'] for n in to_remove])
            leftmost = f.descendents[min(removee_idxs)]
            entry_point = leftmost # where to substitute g into f
            changed = True
            # find the lowest entry point that has all to_removes as
            # descendents
            while changed:
                if entry_point.parent is None:
                    break
                changed = False
                for tr in to_remove:
                    if tr not in entry_point.descendents:
                        changed = True
                        entry_point = entry_point.parent
                        break
            if entry_point.sem_cats in [{'S|NP'}, {'S|NP|NP'}] and len(entry_point.parent.children) == len(entry_point.parent.var_descendents)+1:
                print('stepping up to get vars')
                entry_point = entry_point.parent
            if entry_point.node_type == 'Q':
                g = self.spawn_self_like(' '.join([maybe_brac(c.lf_str,sep=' ')
                    for c in entry_point.children]),idx_in_tree=entry_point.idx_in_tree)
                assert all([c1.lf_str==c2.lf_str for c1,c2 in zip(g.descendents,entry_point.descendents)][1:])
                assert entry_point.descendents[0].lf_str != g.descendents[0].lf_str
            else:
                g = entry_point.copy()
            to_present_as_args_to_g = '' if len(to_remove)==1 and to_remove[0].node_type=='detnoun' else [d for d in entry_point.leaf_descendents if d not in to_remove]
            have_embedded_binders = [d for d in to_present_as_args_to_g if d.node_type=='bound_var' and d.binder in entry_point.descendents]
            #if len(have_embedded_binders) > 0 and self.parent == 'START':
                #print(removee_idxs)
                #print('found vars with embedded binders')
            to_present_as_args_to_g = [x for x in to_present_as_args_to_g if x not in have_embedded_binders]
            assert len(to_present_as_args_to_g) == len(entry_point.leaf_descendents) - n_removees - len(have_embedded_binders)
            g = g.turn_nodes_to_vars(to_present_as_args_to_g)
            if (g.n_lambda_binders > 4) or (strip_string(g.subtree_string().replace(' AND','')) == ''): # don't consider only variables
                continue
            g_sub_var_num = self.new_var_num
            new_entry_point_in_f_as_str = ' '.join([f'${g_sub_var_num}'] + list(reversed([maybe_brac(n.string,sep=' ') for n in to_present_as_args_to_g])))
            assert entry_point.subtree_string() in self.subtree_string()
            entry_point.__init__(new_entry_point_in_f_as_str,self.idx_in_tree,entry_point.cache)
            if len(to_present_as_args_to_g) >= 3: # then f will end up with arity 4
                continue
            if not f.set_cats_from_string():
                continue
            f = f.lambda_abstract(g_sub_var_num)
            self.add_split(f,g,'app')
            # if self is a lambda and had it's bound var removed and put in g then try cmp
            if self.node_type == 'lmbda' and f.node_type == 'lmbda' and any(cat_components(x,allow_atomic=True)[-1] == cat_components(y,allow_atomic=True)[-1] and cat_components(y,allow_atomic=True)[-1]!='X' for x in self.sem_cats for y in f.sem_cats):
                f_cmp_string = logical_type_raise(g.lf_str)
                assert f_cmp_string == logical_type_raise(g.lf_str)
                f_cmp = LogicalForm(f_cmp_string)
                fparts = split_respecting_brackets(f.subtree_string(),sep='.')
                # g_cmp is just g reordered, move the part at n_removees position to front
                # because first var will get chopped off from cmp and need
                # remainder to be able to receive body of f_cmp just like f received g
                #assert bool(re.match(r'lambda \$\d{1,2}',fparts[n_removees]))
                #g_cmp_parts = [fparts[f.n_lambda_binders-1]] + fparts[:f.n_lambda_binders-1] + fparts[f.n_lambda_binders:]
                g_cmp_parts = fparts[1:2] + fparts[:1] + fparts[2:]
                g_cmp_string = '.'.join(g_cmp_parts)
                g_cmp = LogicalForm(g_cmp_string)
                if not (g_cmp.sem_cat_is_set and all(is_atomic(gcsc) for gcsc in g_cmp.sem_cats)): # g_cmp must have slash
                    self.add_split(f_cmp,g_cmp,'cmp')
        if '|' in self.sem_cats:
            assert self.subtree_string().startswith('lambda')
        #self.cache['splits'][self.lf_str] = self.possible_app_splits
        self.cache[self.lf_str] = self.possible_app_splits, self.possible_cmp_splits, self.sem_cats
        return self.possible_app_splits, self.possible_cmp_splits

    def add_split(self,f,g,split_type):
        f.parent = g.parent = self
        if not g.set_cats_from_string():
            if g.lf_str == 'pro:per|you_1':
                breakpoint()
            return
        if all(gsc not in ['X','N'] and len(re.findall(r'[\\/\|]',gsc)) != g.n_lambda_binders for gsc in g.sem_cats):
            #print('discarding', g.sem_cats, g.lf_str)
            return
        if g.sem_cats == {'NP|N'}:
            breakpoint()
        if not g.cat_consistent():
            return
        assert f != self
        assert g != self
        if split_type == 'app':
            if not f.set_cats_from_string():
            #if all(fsc not in ['X','N'] and len(re.findall(r'[\\/\|]',fsc)) != f.n_lambda_binders for fsc in f.sem_cats):
                #print(f.lf_str, f.sem_cats,'line246')
                return
            to_add_to = self.possible_app_splits
        else:
            assert split_type == 'cmp'
            assert f.subtree_string().startswith('lambda')
            assert g.subtree_string().startswith('lambda')
            f.sem_cats = set(['X'])
            to_add_to = self.possible_cmp_splits
        if not combine_lfs(f.subtree_string(),g.subtree_string(),split_type,normalize=True) == self.subtree_string(alpha_normalized=True):
            breakpoint()
        g.infer_splits()
        f.infer_splits()
        if self.sem_cat_is_set and g.sem_cat_is_set:
            self.set_f_sem_cat_from_self_and_g(f,g,split_type)
            if not f.cat_consistent():
                return
        if f.sem_cats.intersection(['S|N','S|NP|N']):
            breakpoint()
            return
        #if f.lf_str == 'lambda $1.lambda $0.not (mod|will_2 ($1 $0))' and g.lf_str == 'lambda $1.v|eat_4 $1':
            #breakpoint()
        if f.lf_str== 'lambda $2.not (mod|will_2 $2)':
            breakpoint()
        if g.sem_cat_is_set and f.sem_cat_is_set and split_type=='app': # no cmp inferred cats ftm
            inferred_sem_cats = set([])
            for fsc in f.sem_cats:
                for gsc in g.sem_cats:
                    #hits = re.findall(fr'[\\/\|]\(?{re.escape(gsc)}\)?$',re.escape(fsc))
                    # it seems '\' needs to be escaped but | needs to NOT be escaped
                    hits = re.findall(fr'[\\/\|]\(?{re.escape(gsc)}\)?$',fsc.replace('\\','\\\\'))
                    assert len(hits) < 2
                    new_inferred_sem_cats = set(fsc[:-len(hit)] for hit in hits)
                    inferred_sem_cats = inferred_sem_cats | new_inferred_sem_cats
            if inferred_sem_cats:
                if self.sem_cats == {'X'}:
                    self.sem_cats = inferred_sem_cats
                else:
                    new = self.sem_cats.intersection(inferred_sem_cats)
                    if not new:
                        breakpoint()
                    self.sem_cats = new
                if not self.cat_consistent():
                    return
        assert f.sem_cat_is_set or not self.sem_cat_is_set or not g.sem_cat_is_set
        if not f.sem_cat_is_set:
            #print(f.lf_str,' + ',g.lf_str)
            return
        assert not (self.sem_cats == set(['S']) and g.sem_cats == set(['N']))
        f.sibling = g
        g.sibling = f
        if f.is_type_congruent():
            to_add_to.append((f,g))
        #else:
            #print(f,'line304')

    def cat_consistent(self):
        if self.sem_cats.intersection(['S|N','Sq|N']):
            return False
        """Return False if sem_cat says type-raised but lf not of type-raised form."""
        return is_type_raised(self.subtree_string()) == ('S|(S|NP)' in self.sem_cats)

    def set_f_sem_cat_from_self_and_g(self,f,g,comb_type):
        if comb_type == 'app':
            new_inferred_sem_cats = set([f'{ssc}|{maybe_brac(gsc)}' for ssc in self.sem_cats for gsc in g.sem_cats])
            assert not any(x is None for x in new_inferred_sem_cats)
        else:
            assert comb_type == 'cmp'
            new_inferred_sem_cats = set(f_cmp_from_parent_and_g(ssc,gsc,sem_only=True)[0] for ssc in self.sem_cats for gsc in g.sem_cats)
            new_inferred_sem_cats = set(x for x in new_inferred_sem_cats if x is not None)
        assert not any(x.startswith('|') for x in new_inferred_sem_cats)
        f.sem_cats = f.sem_cats.intersection(new_inferred_sem_cats) if f.sem_cat_is_set else new_inferred_sem_cats
        f.cache[f.lf_str] = f.possible_app_splits, f.possible_cmp_splits, f.sem_cats
        assert f.sem_cats
        if new_inferred_sem_cats.intersection(set(['S|N','S|NP|N'])):
            return
        for f1,g1 in f.possible_app_splits:
            if g1.sem_cat_is_set and not (all(is_atomic(gsc) for gsc in g.sem_cats) and comb_type=='cmp'):
                f.set_f_sem_cat_from_self_and_g(f1,g1,comb_type)

    def spawn_child(self,defining_string,sibling_idx):
        """Careful, this means a child in the tree of one logical form, not a possible split."""
        return LogicalForm(defining_string,idx_in_tree=self.idx_in_tree+[sibling_idx],cache=self.cache,parent=self)

    def spawn_self_like(self,defining_string,idx_in_tree=None):
        if idx_in_tree is None:
            idx_in_tree = self.idx_in_tree
        new = LogicalForm(defining_string,idx_in_tree=idx_in_tree,cache=self.cache,parent=self.parent)
        new.sem_cats = self.sem_cats
        return new

    @property
    def stripped_subtree_string(self):
        ss = re.sub(r'(lambda \$\d{1,2})+\.','',self.subtree_string())
        ss = re.sub(r'\$\d{1,2}( )*','',ss).replace('()','')
        # allowed to have trailing bound variables
        return strip_string(ss)

    @property
    def lf_str(self): return self.subtree_string()

    @property
    def lf_str_a(self): return self.subtree_string(alpha_normalized=True)

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
        return new_var_num(self.subtree_string())

    @property
    def n_lambda_binders(self):
        return n_lambda_binders(self.subtree_string())

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
            assert 'PLACE' not in string
            self.stored_alpha_normalized_subtree_string = string
        else:
            assert 'PLACE' not in string
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
            assert is_bracketed(x) or ' ' not in x or self.node_type in ['noun','lmbda','Q','detnoun','barenoun']
            x = maybe_debrac(x)
            if alpha_normalized:
                x = alpha_normalize(x)
        if recompute:
            self.set_subtree_string(as_shell,alpha_normalized=alpha_normalized,string=x)
        return x

    def subtree_string_(self,show_treelike,as_shell,recompute):
        if self.node_type == 'lmbda':
            subtree_string = self.children[0].subtree_string(as_shell=as_shell,show_treelike=show_treelike,recompute=recompute)
            if show_treelike:
                subtree_string = subtree_string.replace('\n\t','\n\t\t')
                x = f'{self.string}\n\t{subtree_string}\n'
            else:
                x = f'{self.string}.{subtree_string}'
        elif self.node_type == 'Q':
            assert len(self.children) == 1
            # debraccing only happens in subtree_string() so child already bracced here
            x = self.children[0].subtree_string_(show_treelike,as_shell,recompute=recompute)
            bracced_if_var = f'({x})' if x.startswith('$') else x
            return f'Q {bracced_if_var}'
        elif self.node_type in ['composite','detnoun']:
            child_trees = [maybe_brac(c.subtree_string_(show_treelike,as_shell,recompute=recompute),sep=[' ']) for c in self.children]
            if show_treelike:
                subtree_string = '\n\t'.join([c.replace('\n\t','\n\t\t') for c in child_trees])
                x = f'{self.string}\n\t{subtree_string}'
            else:
                subtree_string = ' '.join(child_trees)
                x = f'{self.string}({subtree_string})'
        elif (not as_shell) or self.node_type in ['bound_var','unbound_var']:
            x = self.string
        else:
            x = self.node_type

        assert as_shell or 'PLACE' not in x
        assert not x.startswith('.')

        return x

    def copy(self):
        copied_version = self.spawn_self_like(self.subtree_string())
        assert copied_version.lf_str_a == self.lf_str_a
        copied_version.sem_cats = self.sem_cats
        return copied_version

    def __repr__(self):
        return (f'LogicalForm of type {self.node_type.upper()}: {self.subtree_string()}'
                f'\n\tsemantic category: {self.sem_cats}\tindex in tree: {self.idx_in_tree}\n')

    def __hash__(self):
        return hash(self.subtree_string(alpha_normalized=True)+str(self.idx_in_tree))

    def __eq__(self,other):
        if not isinstance(other,LogicalForm):
            return False
        return other.lf_str == self.lf_str and other.idx_in_tree == self.idx_in_tree

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
        copied = self.spawn_self_like(lambda_prefix+'.' + copied.subtree_string(),idx_in_tree=[])
        return copied

    def lambda_abstract(self,var_num=None):
        """Returns a new LogicalForm which is the same as self except with 'lambda x' in front."""
        if var_num is None:
            var_num = self.new_var_num
        new = self.spawn_self_like(f'lambda ${var_num}.',idx_in_tree=[])
        new.node_type = 'lmbda'
        new.children = [self]
        assert f'${var_num}' in self.subtree_string()
        new.stored_subtree_string = f'lambda ${var_num}.{self.subtree_string()}'
        new.stored_alpha_normalized_subtree_string =alpha_normalize(f'lambda ${var_num}.{self.subtree_string()}')
        new.stored_shell_subtree_string = f'lambda ${var_num}.{self.subtree_string(as_shell=True)}'
        new.stored_alpha_normalized_shell_subtree_string = alpha_normalize(f'lambda ${var_num}.{self.subtree_string(as_shell=True)}')
        new.var_descendents = list(set(self.var_descendents + [var_num]))
        return new

    @property
    def sem_cat_is_set(self):
        return self.sem_cats != set(['X'])

class ParseNode():
    def __init__(self,lf,words,node_type,parent=None,sibling=None,syn_cats=None):
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
        self.sem_cats = self.logical_form.sem_cats
        if syn_cats is not None:
            self.syn_cats = syn_cats
        else:
            self.syn_cats = lf.sem_cats
        self.is_leaf = self.logical_form.is_semantic_leaf or len(self.words) == 1
        if not self.is_leaf:
            if self.logical_form.possible_app_splits == []:
                self.logical_form.infer_splits()
            for split_point in range(1,len(self.words)):
                left_words = self.words[:split_point]
                right_words = self.words[split_point:]
                for f,g in self.logical_form.possible_app_splits:
                    if g.sem_cat_is_set:
                        self.add_app_splits(f,g,left_words,right_words)
                for f,g in self.logical_form.possible_cmp_splits:
                    if g.sem_cat_is_set:
                        self.add_cmp_splits(f,g,left_words,right_words)

        if len(self.possible_splits) == 0:
            self.is_leaf = True
        assert not any(x is None for x in self.sem_cats)
        assert not any(x is None for x in self.syn_cats)

    def add_cmp_splits(self,f,g,left_words,right_words):
        # just fwd_cmp for now
        right_child_fwd = ParseNode(g,right_words,parent=self,node_type='right_fwd_cmp')
        inferred_f_syn_cats = set([])
        inferred_g_syn_cats = set([])
        for ssync in self.syn_cats:
            for rcsync in right_child_fwd.syn_cats:
                new_inferred_f_syn_cat, new_inferred_g_syn_cat = f_cmp_from_parent_and_g(ssync,rcsync,sem_only=False)
                if new_inferred_f_syn_cat: inferred_f_syn_cats.add(new_inferred_f_syn_cat)
                if new_inferred_g_syn_cat: inferred_g_syn_cats.add(new_inferred_g_syn_cat)

        assert 'S/NP\\NP' not in inferred_f_syn_cats
        if None in inferred_f_syn_cats:
            breakpoint()
        if inferred_f_syn_cats == set([]) or inferred_f_syn_cats == {None}:
            return
        left_child_fwd = ParseNode(f,left_words,parent=self,node_type='left_fwd_cmp',syn_cats=inferred_f_syn_cats)
        if None in inferred_f_syn_cats:
            breakpoint()
        right_child_fwd.syn_cats = set([s for s in right_child_fwd.syn_cats if any(is_congruent(s,y) for y in inferred_f_syn_cats)])
        if None in f.sem_cats:
            breakpoint()
        assert set_congruent(f.sem_cats,inferred_f_syn_cats)
        assert set_congruent(g.sem_cats,inferred_g_syn_cats)
        right_child_fwd.syn_cats = inferred_g_syn_cats
        self.append_split(left_child_fwd,right_child_fwd,'fwd_cmp')

    def add_app_splits(self,f,g,left_words,right_words):
            # CONVENTION: f-child with the words on the left of sentence is
            # the 'left' child, even if bck application, left child is g in
            # that case
            right_child_fwd = ParseNode(g,right_words,parent=self,node_type='right_fwd_app')
            new_syn_cats = set(f'{ssync}/{maybe_brac(rcsync)}' for ssync in self.syn_cats for rcsync in right_child_fwd.syn_cats)
            assert new_syn_cats != 'S/NP\\NP'
            left_child_fwd = ParseNode(f,left_words,parent=self,node_type='left_fwd_app',syn_cats=new_syn_cats)
            #congs = set(x for x in f.sem_cats if any(is_congruent(x,y) for y in new_syn_cats))
            assert set_congruent(f.sem_cats, new_syn_cats)
            self.append_split(left_child_fwd,right_child_fwd,'fwd_app')

            left_child_bck = ParseNode(g,left_words,parent=self,node_type='left_bck_app')
            new_syn_cats = set(f'{ssync}\\{maybe_brac(lcsync)}' for ssync in self.syn_cats for lcsync in left_child_bck.syn_cats if not (ssync=='S/NP' and lcsync=='NP'))
            if new_syn_cats:
                assert set_congruent(f.sem_cats, new_syn_cats)
                right_child_bck = ParseNode(f,right_words,parent=self,node_type='right_bck_app',syn_cats=new_syn_cats)
                self.append_split(left_child_bck,right_child_bck,'bck_app')

    def append_split(self,left,right,combinator):
        left.siblingify(right)
        assert left.sibling is not None
        assert right.sibling is not None
        self.possible_splits.append({'left':left,'right':right,'combinator':combinator})

    @property
    def is_g(self):
        return self.node_type in ['right_fwd_app','left_bck_app','right_fwd_cmp','left_bck_cmp']

    @property
    def is_fwd(self):
        return self.node_type in ['right_fwd_app','left_fwd_app','right_fwd_cmp','left_fwd_cmp']

    @property
    def lf_str(self):
        return self.logical_form.subtree_string(alpha_normalized=True)

    def info_if_leaf(self):
        shell_lf = self.logical_form.subtree_string(as_shell=True,alpha_normalized=True)
        lf = self.logical_form.subtree_string(alpha_normalized=True)
        word_str = ' '.join(self.words)
        sem_cats = set(maybe_de_type_raise(ssc) for ssc in self.sem_cats)
        syn_cats = set(maybe_de_type_raise(ssc) for ssc in self.syn_cats)
        if sem_cats != self.sem_cats: # has been de-type-raised
            assert syn_cats != self.syn_cats
            assert all(any(maybe_de_type_raise(s1)==s2 for s1 in self.sem_cats) for s2 in sem_cats)
            assert all(any(maybe_de_type_raise(s1)==s2 for s1 in self.syn_cats) for s2 in syn_cats)
            shell_lf = alpha_normalize(logical_de_type_raise(shell_lf))
            lf = alpha_normalize(logical_de_type_raise(lf))
        return word_str, lf, shell_lf, sem_cats, syn_cats

    def __repr__(self):
        base = (f"ParseNode\n"
                f"\tWords: {' '.join(self.words)}\n"
                f"\tLogical Form: {self.logical_form.subtree_string()}\n"
                f"\tSyntactic Category: {self.syn_cats}\n")
        if hasattr(self,'stored_prob'):
            base += f'\tProb: {self.stored_prob}\n'
        return base

    def __eq__(self,other):
        if not isinstance(other,LogicalForm):
            return False
        return self.lf_str == other.lf_str and self.words == other.words

    def __hash__(self):
        return hash(self.lf_str + ' '.join(self.words))

    def siblingify(self,other):
        self.sibling = other
        other.sibling = self

    def propagate_below_probs(self,syntax_learner,shell_meaning_learner,meaning_learner,word_learner,cache,split_prob,is_map):
        self.split_prob = split_prob
        if self in cache:
            return cache[self]
        all_probs = [self.prob_as_leaf(syntax_learner,shell_meaning_learner,meaning_learner,word_learner)]
        for ps in self.possible_splits:
            #syntax_split = ps['left'].syn_cat + ' + ' + ps['right'].syn_cat
            #split_prob = syntax_learner.prob(syntax_split,self.syn_cats)
            split_prob = max(syntax_learner.prob(f'{psl} + {psr}',ss) for psl in ps['left'].syn_cats for psr in ps['right'].syn_cats for ss in self.syn_cats)
            if split_prob == 0:
                breakpoint()
            left_below_prob = ps['left'].propagate_below_probs(syntax_learner,shell_meaning_learner,meaning_learner,word_learner,cache,split_prob,is_map)
            right_below_prob = ps['right'].propagate_below_probs(syntax_learner,shell_meaning_learner,meaning_learner,word_learner,cache,split_prob,is_map)
            all_probs.append(left_below_prob*right_below_prob*split_prob)

        below_prob = max(all_probs) if is_map else sum(all_probs)
        if below_prob == 0:
            breakpoint()
        cache[self] = below_prob
        self.below_prob = below_prob
        return below_prob

    def propagate_above_probs(self,passed_above_prob): #reuse split_prob from propagate_below_probs
        self.above_prob = passed_above_prob*self.split_prob
        for ps in self.possible_splits:
            # this is how to get cousins probs
            #if ps['left'].syn_cats == 'S/N' and ps['right'].syn_cat == 'N':
                #breakpoint()
            ps['left'].propagate_above_probs(self.above_prob*ps['right'].below_prob)
            ps['right'].propagate_above_probs(self.above_prob*ps['left'].below_prob)
            assert np.allclose(ps['left'].above_prob*ps['left'].below_prob, ps['right'].above_prob*ps['right'].below_prob)
        if self.possible_splits != []:
            if not self.above_prob > max([z.above_prob for ps in self.possible_splits for z in (ps['right'],ps['left'])]):
                breakpoint()

    def prob_as_leaf(self,syntax_learner,shell_meaning_learner,meaning_learner,word_learner):
        word_str, lf, shell_lf, sem_cats, syn_cats = self.info_if_leaf()
        #word_str, lf, shell_lf, = self.info_if_leaf()
        for sc in sem_cats:
            assert any(is_congruent(ssync,sc) for ssync in syn_cats)
        self.stored_prob_as_leaf = max(syntax_learner.prob('leaf',ssync) for ssync in syn_cats) * \
                                   max(shell_meaning_learner.prob(shell_lf,sc) for sc in sem_cats) * \
                                   meaning_learner.prob(lf,shell_lf) * \
                                   word_learner.prob(word_str,lf)
                            # will use stored value in train_one_step()
        if self.stored_prob_as_leaf==0:
            breakpoint()
        return self.stored_prob_as_leaf
