import numpy as np
from functools import partial
from utils import split_respecting_brackets, is_bracketed, maybe_brac, is_atomic, strip_string, cat_components, is_congruent, alpha_normalize, maybe_debrac, f_cmp_from_parent_and_g, combine_lfs, logical_type_raise, is_wellformed_lf, new_var_num, n_lambda_binders, set_congruent, lf_cat_congruent, lambda_match, is_bracket_balanced, apply_sem_cats, parent_cmp_from_f_and_g, balanced_substrings, non_directional, base_cats_from_str, add_q, de_q, possible_syncs
from errors import SemCatError, ZeroProbError, SynCatError
import re
import sys; sys.setrecursionlimit(500)
from learner_config import pos_marking_dict, chiltag_to_node_types


# is_leaf means it's atomic in lambda calculus
# is_semantic_leaf means we shouldn't consider breaking it further
# e.g. lambda $0.runs $0 is a semantic leaf but not a leaf

# split_prob is the prob, according to the syntax_learner, of the split that
# gave birth to it; above_prob is the prob of the branch this node is on, equal
# to split_prob*parent.above_prob*sibling.below_prob; below_prob is the prob of
# this node, under all possible ways of splitting it down to the leaves;

# Q(.) is like a composite, or composite is like a dummy wrapper around an lf,
# Q then being an alternative non-dummy wrapper

debug_split_lf = None
debug_set_cats = None


class LogicalForm:
    def __init__(self,defining_string,idx_in_tree=[],caches={'splits':{},'cats':{}},parent=None,dblfs=None,dbsss=None, verbose_as=False, cats_from_traised=None):
        self.verbose_as = verbose_as
        if dblfs is not None:
            global debug_split_lf
            assert debug_split_lf in (None, dblfs)
            debug_split_lf = dblfs
        if dbsss is not None:
            global debug_set_cats
            assert debug_set_cats in (None, dbsss)
            debug_set_cats = dbsss
        assert isinstance(idx_in_tree,list)
        assert is_wellformed_lf(defining_string)
        had_surrounding_brackets = False
        self.sibling = None
        self.idx_in_tree = idx_in_tree
        self.caches = caches
        self.var_descs = []
        self.app_splits = set()
        self.cmp_splits = set()
        self.parent = parent #None if logical root, tmp if will be reassigned, e.g. in infer_splits
        self.stored_subtree_string = ''
        self.stored_shell_subtree_string = ''
        self.stored_alpha_normalized_subtree_string = ''
        self.stored_alpha_normalized_shell_subtree_string = ''
        self.ud_pos = ''
        self.was_cached = False
        self.cats_from_traised = cats_from_traised
        defining_string = maybe_debrac(defining_string)
        if parent == 'START':
            self.sem_cats = set('X')
        if defining_string.startswith('BARE'):
            print('\n'+defining_string+'\n')
            raise SemCatError()
        if defining_string=='not':
            self.node_type = 'neg'
            self.string = 'not'
            self.is_leaf = True
        elif bool(lambda_match(defining_string)):
            lambda_string, _, remaining_string = defining_string.partition('.')
            variable_index = lambda_string.partition('_')[0][7:]
            self.is_leaf = False
            self.string = lambda_string
            self.node_type = 'lmbda'
            self.children = [self.spawn_child(remaining_string,0)]

            for d in self.descs:
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
            assert not any(a==defining_string for a in arguments)
            assert arguments != [self] # would enter a loop if so
            if len(arguments) == 1 and is_bracketed(defining_string):
                had_surrounding_brackets = True
                assert not is_bracketed(defining_string[1:-1])
                #arguments = split_respecting_brackets(defining_string[1:-1])
            #left = ' '.join(arguments[:-1])
            #right = arguments[-1]
            self.children = [self.spawn_child(a,i) for i,a in enumerate(arguments)]
            #self.children = [self.spawn_child(a,i) for i,a in enumerate([left, right])]
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
                self.extend_var_descs(int(self.string[1:]))
            elif self.string == 'and':
                self.node_type = 'connective'
            elif self.string == 'prog':
                self.node_type = 'prog'
            elif pos_marking_dict.get(defining_string,None) == 'N':
                breakpoint()
                self.node_type = 'noun'
            elif defining_string.split('|')[0] == 'mod':
                self.node_type = 'raise'
            elif (sds:=strip_string(defining_string)) in ['WHAT', 'WHO']:
                self.node_type = 'WH'
            elif sds in ['you', 'equals', 'hasproperty']:
                self.node_type = sds
            elif sds == '':
                self.node_type = 'null'
            elif re.match(r'[\w:]+\|', defining_string):
                chiltag = defining_string.split('|')[0]
                if chiltag=='n' and defining_string.endswith('BARE'):
                    self.node_type = 'entity'
                elif chiltag=='n:prop' and defining_string.endswith('\'s'):
                    self.node_type = 'quant'
                elif chiltag not in chiltag_to_node_types:
                    if chiltag not in ['part','aux','on','poss','pro:exist','post']:
                        print(defining_string)
                    self.node_type = chiltag
                else:
                    self.node_type = chiltag_to_node_types[chiltag]
            else:
                breakpoint()

        if self.is_leaf: # should only have children defined if comes from the __init__ in splits
            self.children = []

        if had_surrounding_brackets:
            assert self.subtree_string() == defining_string
        else:
            if not self.subtree_string() == defining_string:
                breakpoint()

        self.set_cats()

    def set_cats(self):
        if self.cats_from_traised is not None:
            self.sem_cats = self.cats_from_traised
            self.is_semantic_leaf = True
        elif self.parent == 'START':
            self.is_semantic_leaf = False
        elif self.lf_str in self.caches['cats']:
            self.sem_cats, self.is_semantic_leaf = self.caches['cats'][self.lf_str]
            return self.sem_cats
        else:
            if debug_set_cats is not None and debug_set_cats==self.subtree_string(recompute=True):
                breakpoint()
            self.sem_cats, self.is_semantic_leaf = base_cats_from_str(self.lf_str)
        assert self.sem_cats!=''
        assert self.sem_cats is not None
        assert not any(sc is None for sc in self.sem_cats)
        assert isinstance(self.sem_cats,set)
        #if self.sem_cats == {'NP'} and self.subtree_string(as_shell=True) == 'lambda $0.$0 entity':
            #self.sem_cats = {'S|(S|NP)'}
        if not (self.parent=='START'):
            self.caches['cats'][self.lf_str] = self.sem_cats, self.is_semantic_leaf#, is_congruent
        return self.sem_cats

    def is_cat_congruent(self):
        assert ' you' not in self.lf_str
        self.sem_cats = set(ssc for ssc in self.sem_cats if ssc=='X' or lf_cat_congruent(self.lf_str,ssc))
        return self.sem_cats != set([])

    def infer_splits(self):
        if self.is_semantic_leaf or self.n_lambda_binders >= 3:
            self.app_splits = set()
            self.cmp_splits = set()
            return self.app_splits
        if self.lf_str in self.caches['splits']:
            self.app_splits, self.cmp_splits, self.sem_cats = self.caches['splits'][self.lf_str]
            self.was_cached = True
            return self.app_splits
        if debug_split_lf is not None and debug_split_lf == self.subtree_string(recompute=True):
            breakpoint()
        if self.lf_str.startswith('lambda'):
            maybe_cmps = [x for x in balanced_substrings(self.lf_str) if f'${self.var_num}' in x and strip_string(x)!='']
            for mc in maybe_cmps:
                if re.match(fr'(lambda \$\d{{1,2}}\.)*(Q )?{re.escape(mc)}', self.lf_str):
                    continue
                lmbdas_to_include = sorted(re.findall(r'\$\d{1,2}', mc), key=lambda x:self.lf_str.index(x))
                assert all('lambda '+x in self.lf_str.rpartition('.')[0] for x in lmbdas_to_include)
                g_lambda_terms = '.'.join(f'lambda {x}' for x in lmbdas_to_include)
                gcmp = LogicalForm(f'{g_lambda_terms}.{maybe_debrac(mc)}', caches=self.caches)
                if all(gsc!='X' and is_atomic(gsc) for gsc in gcmp.sem_cats): # this may be blocking traising
                    continue
                if gcmp.sem_cats == {'NP|N'}: # don't compose with determiners
                    continue
                fghole = lmbdas_to_include[0] if len(lmbdas_to_include)==1 else '('+' '.join(lmbdas_to_include)+')'
                f_cmp_string = self.lf_str.replace(mc, fghole)
                if strip_string(f_cmp_string) == 'Q':
                    continue # no just Q nodes for now
                fcmp = LogicalForm(f_cmp_string, caches=self.caches)
                if not fcmp.set_cats():
                    continue
                if (gcmp.n_lambda_binders > 4) or (gcmp.stripped_subtree_string)=='': # don't consider only variables
                    breakpoint()
                    continue
                self.add_split(fcmp, gcmp, 'cmp')
        has_q = any(x.node_type=='Q' for x in self.descs)
        #if has_q:=any(x.node_type=='Q' for x in self.descs):
        #    qidx = [i for i,x in enumerate(self.descs) if x.node_type=='Q'][0]
        #    head_order = ['wh', 'raise', 'hasproperty', 'cop', 'verb', 'part']
        #    head_options = sorted(self.leaf_descs, key=lambda x: head_order.index(x.node_type))
        #    if len(head_options)==0:
        #        head_options = [x for x in self.leaf_descs if x.node_type in ['prep','quant','adv','aux']]
        #    if len(head_options)==0:
        #        self.app_splits = set()
        #        self.cmp_splits = set()
        #        return self.app_splits
        #    else:
        #        head = head_options[0]

        remove_types = ['det', 'noun', 'quant'] if self.node_type=='detnoun' else ['entity', 'composite', 'WH', 'adj', 'prep', 'detnoun']
        pridxs = [i for i,d in enumerate(self.descs) if d.node_type in remove_types]
        cns = [x for x in self.leaf_descs if x.node_type not in ['lmbda', 'composite', 'bound_var', 'unbound_var']]
        if debug_split_lf is not None and debug_split_lf == self.subtree_string(recompute=True):
            breakpoint()
        for ridx in pridxs:
            f = self.copy()
            removee = f.descs[ridx]
            fcns = [x for x in removee.leaf_descs if x.node_type not in ['lmbda', 'composite', 'bound_var', 'unbound_var']]
            if len(fcns) == len(cns):
                continue
            g = removee.copy()
            # lambdas are abstracted in reversed order, need to make it end up
            # the same order as in self
            ordered_g_var_descs = sorted(g.var_descs, key=lambda x: self.lf_str.index(str(x)), reverse=True)
            for v in ordered_g_var_descs:
                g = g.lambda_abstract(v)

            if (g.n_lambda_binders > 4) or (strip_string(g.subtree_string().replace(' AND','')) == ''): # don't consider only variables
                continue
            assert removee.subtree_string() in self.subtree_string()
            if any(x is y for x in self.descs for y in f.descs):
                breakpoint()
            # here we place variables in the same order as they are eaten up by self
            new_entry_point_in_f_as_str = ' '.join([f'${self.new_var_num}'] + list([f'${v}' for v in reversed(ordered_g_var_descs)]))
            removee.__init__(new_entry_point_in_f_as_str, self.idx_in_tree, removee.caches)
            f = f.lambda_abstract(self.new_var_num)
            x,_,y = f.lf_str.rpartition('.')
            if set(re.findall(r'\$\d',x)) != set(re.findall(r'\$\d',y)):
                breakpoint()
            if ('mod|' in f.lf_str or 'prog' in f.lf_str) and g.is_leaf and g.lf_str.startswith('v|'):
                f.infer_splits()
                if any(sc in f.sem_cats for sc in ['S|NP|(S|NP)', 'S|(S|NP)', 'Sq|(S|NP)']):
                    if re.search(fr'[\.\(]{re.escape(g.lf_str)}', self.lf_str):
                        g = g.spawn_self_like(f'lambda $0.{g.subtree_string()} $0')
            self.add_split(f, g, 'app')
            if g.node_type in ('entity', 'WH'):
                combine_opts = set([can_compose_traised(fsc, gsc) for fsc in f.sem_cats for gsc in g.sem_cats])
                if combine_opts == {'no'}:
                    continue
                traised_gstr = de_q(f.lf_str)
                traised_fstr = logical_type_raise(g.lf_str)
                if has_q:
                    traised_fstr = add_q(traised_fstr)
                if 'app' in combine_opts:
                    f_cats = set(f'{sc}|{maybe_brac(fc)}' for sc in self.sem_cats for fc in f.sem_cats)
                    traised_f = LogicalForm(traised_fstr, caches=self.caches, cats_from_traised=f_cats)
                    traised_g = LogicalForm(traised_gstr, caches=self.caches)
                    self.add_split(traised_f, traised_g, 'app')
                if 'cmp' in combine_opts:
                    f_cats = set()
                    for fsc in f.sem_cats:
                        ccs = split_respecting_brackets(fsc, '|')
                        assert len(ccs)==3
                        f_cats = f_cats | {f'{ccs[0]}|({ccs[0]}|{ccs[1]})'}
                    traised_f = LogicalForm(traised_fstr, caches=self.caches, cats_from_traised=f_cats)
                    traised_g_parts = traised_gstr.split('.')
                    if not ( len(traised_g_parts) == 3):
                        print(f"not traising+cmp for {f.lf_str} & {g.lf_str} cuz not enough lmbdas")
                        continue
                    traised_gstr = '.'.join([traised_g_parts[i] for i in [1,0,2]])
                    traised_g = LogicalForm(traised_gstr, caches=self.caches)
                    self.add_split(traised_f, traised_g, 'cmp')
        if '|' in self.sem_cats:
            assert self.subtree_string().startswith('lambda')
        if self.sem_cats == set():
            breakpoint()
            raise SemCatError('empty semcats after inferring splits')
        self.put_self_in_caches()
        if any(len(x.sem_cats)==0 for s in self.splits for x in s):
            breakpoint()
        return self.app_splits, self.cmp_splits

    def add_split(self,f,g,split_type):
        f.parent = g.parent = self
        if self.verbose_as:
            print(f'trying to add {split_type}split for\n{self}\n{f} and\n{g}')
        if not g.set_cats():
            return
        assert f != self
        assert g != self
        if split_type == 'cmp':
            assert f.subtree_string().startswith('lambda')
            assert g.subtree_string().startswith('lambda')
        fstrcats = f.set_cats()
        gstrcats = g.set_cats()
        g.infer_splits()
        if any(len(s) > 20 for s in g.sem_cats):
            if self.verbose_as:
                print('skipping because g got a v long sem_cat')
            return
        if len(split_respecting_brackets(g.lf_str)) == 2 and g.lf_str.startswith('v|') and g.sem_cats=={'S'}:
            g.sem_cats = {'S','S|NP'} # hack to make sure it considers S|NP for VPs
        f.infer_splits() # don't do again if already done in infer_splits
        if any(len(s) > 20 for s in f.sem_cats):
            if self.verbose_as:
                print('skipping because f got a v long sem_cat')
            return
        if f.sem_cats.intersection(['S|N','S|NP|N']):
            return
        comb_fn = apply_sem_cats if split_type=='app' else partial(parent_cmp_from_f_and_g, sem_only=True)
        fcats_to_use = f.sem_cats if split_type=='app' else set(c for c in f.sem_cats if '|' in c)
        gcats_to_use = g.sem_cats if split_type=='app' else set(c for c in g.sem_cats if '|' in c)
        inferred_sem_cats = set([])
        if g.sem_cat_is_set and f.sem_cat_is_set:
            for fsc in fcats_to_use:
                for gsc in gcats_to_use:
                    new_inferred_sem_cat = comb_fn(fsc, gsc)
                    if new_inferred_sem_cat is not None:
                        new_inferred_sem_cat = maybe_debrac(new_inferred_sem_cat)
                        inferred_sem_cats = inferred_sem_cats | set([new_inferred_sem_cat])
        if len(inferred_sem_cats) == 0:
            if self.verbose_as:
                print('skipping because no inferred sem_cats')
            return
        else:
            if self.sem_cats == {'X'}:
                self.sem_cats = inferred_sem_cats
            else:
                self.sem_cats = self.sem_cats.union(inferred_sem_cats)
                self.set_f_sem_cat_from_self_and_g(f,g,split_type)
            g.sem_cats = set([gsc for gsc in gcats_to_use if 'X' in gsc or any(comb_fn(fsc, gsc)==sc for fsc in fcats_to_use for sc in self.sem_cats)])
            f.sem_cats = set([fsc for fsc in fcats_to_use if 'X' in fsc or any(comb_fn(fsc, gsc)==sc for gsc in gcats_to_use for sc in self.sem_cats)])
            g.sem_cats = set([gsc for gsc in g.sem_cats if lf_cat_congruent(g.lf_str,gsc)])
            f.sem_cats = set([fsc for fsc in f.sem_cats if lf_cat_congruent(f.lf_str,fsc)])
        if 'qn|' not in self.lf_str:
            if 'NP|NP' in f.sem_cats:
                breakpoint()
            if 'NP|NP' in g.sem_cats:
                breakpoint()
        if len(f.sem_cats)==0:
            if self.verbose_as:
                print('skipping because f has no semcats')
            return
        if len(g.sem_cats)==0:
            if self.verbose_as:
                print('skipping because f has no semcats')
            return
        if not f.sem_cat_is_set:
            if self.verbose_as:
                print('skipping because f semcat not set')
            return
        assert (self.sem_cats != set(['S']) or g.sem_cats != set(['N']))
        assert ( combine_lfs(f.subtree_string(),g.subtree_string(),split_type,normalize=True) == self.subtree_string(alpha_normalized=True))
        f.sibling = g
        g.sibling = f
        f.put_self_in_caches()
        g.put_self_in_caches()
        if len(f.sem_cats) == 0 or len(g.sem_cats)==0:
            breakpoint()
        if split_type=='app':
            self.app_splits = self.app_splits | {(f,g)}
        else:
            assert split_type == 'cmp'
            self.cmp_splits = self.cmp_splits | {(f,g)}
        if not ( self.sem_cats != set()):
            breakpoint()

    def put_self_in_caches(self):
        existing_app_splits, existin_cmp_splits, existing_sem_cats = self.caches['splits'].get(self.lf_str, ({},{},{}))
        self.caches['splits'][self.lf_str] = self.app_splits.union(existing_app_splits), self.cmp_splits.union(existin_cmp_splits), self.sem_cats.union(existing_sem_cats)

    def set_f_sem_cat_from_self_and_g(self,f,g,comb_type):
        if comb_type == 'app':
            new_inferred_sem_cats = set([f'{ssc}|{maybe_brac(gsc)}' for ssc in self.sem_cats for gsc in g.sem_cats])
            assert not any(x is None for x in new_inferred_sem_cats)
        else:
            assert comb_type == 'cmp'
            new_inferred_sem_cats = set(f_cmp_from_parent_and_g(ssc,gsc,sem_only=True)[0] for ssc in self.sem_cats for gsc in g.sem_cats)
            new_inferred_sem_cats = set(x for x in new_inferred_sem_cats if x is not None)
        assert not any(x.startswith('|') for x in new_inferred_sem_cats)
        if new_inferred_sem_cats.intersection(set(['S|N','S|NP|N'])):
            return
        old_fsc = f.sem_cats
        if f.sem_cats | new_inferred_sem_cats in [{'Sq|NP','Sq|(S|NP)'},{'S|NP','S|(S|NP)'}]:
            print(self, f, g)
            f.sem_cats = f.sem_cats | new_inferred_sem_cats
        else:
            f.sem_cats = set_congruent(new_inferred_sem_cats, f.sem_cats)
        if not f.sem_cats:
            if len(old_fsc)>0:
                print(f'fsem_cats were {old_fsc}, inferred are {new_inferred_sem_cats}')
            return
        if f.sem_cats == {'Sq|(S|(S|NP))'}:
            breakpoint()
        for f1,g1 in f.app_splits:
            if g1.sem_cat_is_set and not (all(is_atomic(gsc) for gsc in g.sem_cats) and comb_type=='cmp'):
                f.set_f_sem_cat_from_self_and_g(f1,g1,comb_type)

    def spawn_child(self,defining_string,sibling_idx):
        """Careful, this means a child in the tree of one logical form, not a possible split."""
        return LogicalForm(defining_string,idx_in_tree=self.idx_in_tree+[sibling_idx],caches=self.caches,parent=self, verbose_as=self.verbose_as)

    def spawn_self_like(self,defining_string,idx_in_tree=None):
        if idx_in_tree is None:
            idx_in_tree = self.idx_in_tree
        new = LogicalForm(defining_string,idx_in_tree=idx_in_tree,caches=self.caches,parent=self.parent, verbose_as=self.verbose_as)
        new.sem_cats = self.sem_cats
        return new

    @property
    def stripped_subtree_string(self):
        # allowed to have trailing bound variables
        return strip_string(self.subtree_string())

    @property
    def lf_str(self): return self.subtree_string()

    @property
    def lf_str_a(self): return self.subtree_string(alpha_normalized=True)

    @property
    def descs(self):
        return [self] + [x for item in self.children for x in item.descs]

    @property
    def leaf_descs(self):
        if self.is_leaf:
            return [self]
        return [x for item in self.children for x in item.leaf_descs]

    @property
    def var_num(self):
        assert self.node_type=='lmbda', "you're trying to get a var from a non-lmbda node"
        return int(self.string[8:])

    @property
    def new_var_num(self):
        return new_var_num(self.subtree_string())

    @property
    def n_lambda_binders(self):
        return n_lambda_binders(self.subtree_string())

    @property
    def splits(self):
        return list(self.app_splits) + list(self.cmp_splits)

    def extend_var_descs(self,var_num):
        if var_num not in self.var_descs:
            self.var_descs.append(var_num)
        if self.parent is not None and self.parent != 'START':
            self.parent.extend_var_descs(var_num)

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
            assert is_bracket_balanced(x)
            assert is_bracketed(x) or ' ' not in x or self.node_type=='entity' and x.startswith('BARE') or self.node_type in ['noun','lmbda','Q','detnoun']
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
            return f'Q {maybe_brac(x)}'
        elif self.node_type in ['composite','detnoun']:
            child_trees = [maybe_brac(c.subtree_string_(show_treelike,as_shell,recompute=recompute),sep=[' ']) for c in self.children]
            if self.node_type=='composite' and self.children[0].node_type=='composite':
                assert is_bracketed(child_trees[0])
                child_trees[0] = child_trees[0][1:-1]
                assert not is_bracketed(child_trees[0])
            if show_treelike:
                subtree_string = '\n\t'.join([c.replace('\n\t','\n\t\t') for c in child_trees])
                x = f'{self.string}\n\t{subtree_string}'
            else:
                subtree_string = ' '.join(child_trees)
                x = f'{self.string}({subtree_string})'
        elif (not as_shell) or self.node_type in ['bound_var','unbound_var']:
            x = self.string
        elif self.string.startswith('v|'):
            x = 'vconst'
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
        removees = [d for d in copied.descs if d in nodes]
        # node may be 'in' another list if there is another node in that list
        # that has the same defining string as node, but we want the reference,
        # i.e. node itself, rather than its copy in the list
        vars_abstracted = []
        for desc in removees:
            var_num = copied.new_var_num
            desc.string = f'${var_num}'
            vars_abstracted.append(var_num)
            desc.extend_var_descs(var_num)
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
        assert f'${var_num}' in self.subtree_string(recompute=True)
        new.stored_subtree_string = f'lambda ${var_num}.{self.subtree_string()}'
        new.stored_alpha_normalized_subtree_string =alpha_normalize(f'lambda ${var_num}.{self.subtree_string()}')
        new.stored_shell_subtree_string = f'lambda ${var_num}.{self.subtree_string(as_shell=True)}'
        new.stored_alpha_normalized_shell_subtree_string = alpha_normalize(f'lambda ${var_num}.{self.subtree_string(as_shell=True)}')
        new.var_descs = list(set(self.var_descs + [var_num]))
        return new

    @property
    def sem_cat_is_set(self):
        return self.sem_cats != set(['X'])

class ParseNode():
    def __init__(self, lf, words, node_type, sync, parent=None, sibling=None):
        self.lf = lf
        self.words = words
        self.splits = []
        self.parent = parent
        self.node_type = node_type
        self.sibling = sibling
        assert (parent is None) or (node_type != 'ROOT'), \
            "you're trying to specify a parent for the root node"
        assert (parent is not None) or (node_type == 'ROOT'), \
            "you're failing to specify a parent for a non-root node"
        assert (sibling is None) or (node_type != 'ROOT'), \
            "you're trying to specify a sibling for the root node"
        self.sem_cats = self.lf.sem_cats
        self.sync = sync
        self.is_leaf = self.lf.is_semantic_leaf or len(self.words) == 1
        if not self.is_leaf:
            for split_point in range(1,len(self.words)):
                left_words = self.words[:split_point]
                right_words = self.words[split_point:]
                for f,g in self.lf.app_splits:
                    if not g.sem_cat_is_set:
                        breakpoint()
                    if g.sem_cat_is_set:
                        try:
                            self.add_splits(f,g,left_words,right_words, 'fwd', 'app')
                            self.add_splits(f,g,left_words,right_words, 'bck', 'app')
                        except SynCatError:
                            pass
                for f,g in self.lf.cmp_splits:
                    if not g.sem_cat_is_set:
                        breakpoint()
                    try:
                        self.add_splits(f,g,left_words,right_words, 'fwd', 'cmp')
                        self.add_splits(f,g,left_words,right_words, 'bck', 'cmp')
                    except SynCatError:
                        pass

        if len(self.splits) == 0:
            self.is_leaf = True

    def add_splits(self, f, g, left_words, right_words, direction, split_type):
        # CONVENTION: f-child with the words on the left of sentence is the
        # 'left' child, even if bck application, left child is g in that case
        if len(g.sem_cats) > 1:
            breakpoint()
        if len(g.sem_cats) == 0:
            breakpoint()
        all_g_syncs = [x for s in g.sem_cats for x in possible_syncs(s)]
        for gsc in all_g_syncs:
            if bool(re.match(r'S/[A-Za-z\(\|\)]+\\NP$', gsc)):
                continue
            if '|' in gsc:
                breakpoint()
            if direction=='fwd':
                g_child = ParseNode(g,right_words,parent=self,node_type='right_fwd_app', sync=gsc)
                if split_type == 'app':
                    f_sync = f'{self.sync}/{maybe_brac(gsc)}'
                else:
                    assert split_type == 'cmp'
                    f_sync, g_sync = f_cmp_from_parent_and_g(self.sync, gsc, sem_only=False)
                    if g_sync is None:
                        continue
                    if not is_congruent(g_sync, gsc):
                        breakpoint()
                f_child = ParseNode(f,left_words,parent=self,node_type='left_fwd_app',sync=f_sync)
                self.append_split(f_child, g_child, 'fwd_app')
            else:
                assert direction=='bck'
                g_child = ParseNode(g,left_words,parent=self,node_type='right_bck_app', sync=gsc)
                if split_type == 'app':
                    f_sync = f'{self.sync}\\{maybe_brac(gsc)}'
                    #if bool(re.match(r'S/[A-Za-z]\\NP+$', f_sync)):
                else:
                    assert split_type == 'cmp'
                    f_sync, g_sync = f_cmp_from_parent_and_g(self.sync, gsc, sem_only=False)
                    if g_sync is None:
                        continue
                    if not is_congruent(g_sync, gsc):
                        breakpoint()
                if bool(re.match(r'S/[A-Za-z\(\|\)]+\\NP$', f_sync)):
                    continue
                f_child = ParseNode(f,right_words,parent=self,node_type='left_bck_app',sync=f_sync)
                self.append_split(g_child, f_child, 'bck_app')
            #new_syncs = set(f'{self.sync}/{maybe_brac(rcsync)}' for rcsync in right_child_fwd.syncs)
            assert 'S/NP\\NP' != f_sync
            #bad_new_syncs = [sc for sc in ('X\\S', 'X/S') if sc in new_syncs]
            #if len(bad_new_syncs) > 0:
                #breakpoint()
                #raise SemCatError(f"bad new syn cats: {''.join(bad_new_syncs)}")
            if 'S\\NP/(S|NP)' == f_sync:
                breakpoint()

    def append_split(self,left,right,combinator):
        left.siblingify(right)
        assert left.sibling is not None
        assert right.sibling is not None
        self.splits.append({'left':left,'right':right,'combinator':combinator})

    @property
    def is_g(self):
        return self.node_type in ['right_fwd_app','left_bck_app','right_fwd_cmp','left_bck_cmp']

    @property
    def prob(self):
        return self.above_prob*self.below_prob

    @property
    def is_fwd(self):
        return self.node_type in ['right_fwd_app','left_fwd_app','right_fwd_cmp','left_fwd_cmp']

    @property
    def lf_str(self):
        return self.lf.subtree_string(alpha_normalized=True)

    def __repr__(self):
        base = (f"ParseNode\n"
                f"\tWords: {' '.join(self.words)}\n"
                f"\tLogical Form: {self.lf.subtree_string()}\n"
                f"\tSyntactic Category: {self.sync}\n")
        if hasattr(self,'stored_prob'):
            base += f'\tProb: {self.stored_prob}\n'
        return base

    def show_splits(self):
        for i,split in enumerate(self.splits):
            f, g = split['left'], split['right']
            fwords, gwords = ' '.join(f.words), ' '.join(g.words)
            print(f'{i}: {f.lf_str} {fwords} {f.sync}\t+\t{g.lf_str} {gwords} {g.sync}')

    def __eq__(self,other):
        if not isinstance(other,LogicalForm):
            return False
        return self.lf_str == other.lf_str and self.words == other.words

    def __hash__(self):
        return hash(self.lf_str + ' '.join(self.words))

    def siblingify(self,other):
        self.sibling = other
        other.sibling = self

    def propagate_below_probs(self,syntaxl,shell_meaningl,meaningl,wordl,prob_cache,split_prob,is_map):
        """Recursively called on children, and use results for this node, so probs are propagated up the tree."""
        self.split_prob = split_prob # probability of parent being split this way to produce this
        if self in prob_cache:
            return prob_cache[self]
        all_probs = [('leaf','leaf',self.prob_as_leaf(syntaxl,shell_meaningl,meaningl,wordl))]
        for ps in self.splits:
            #split_prob = max(syntaxl.prob(f'{psl} + {psr}',ss) for psl in ps['left'].syncs for psr in ps['right'].syncs for ss in self.syncs)
            split_prob = syntaxl.prob(f'{ps["left"].sync} + {ps["right"].sync}',self.sync)
            if split_prob == 0:
                breakpoint()
            left_below_prob = ps['left'].propagate_below_probs(syntaxl,shell_meaningl,meaningl,wordl,prob_cache,split_prob,is_map)
            right_below_prob = ps['right'].propagate_below_probs(syntaxl,shell_meaningl,meaningl,wordl,prob_cache,split_prob,is_map)
            all_probs.append((ps['left'], ps['right'], left_below_prob*right_below_prob*split_prob))

        self.best_split = max(all_probs, key=lambda x:x[2])
        self.below_prob = max([x[2] for x in all_probs]) if is_map else sum([x[2] for x in all_probs])
        self.rel_prob = self.below_prob/sum([x[2] for x in all_probs])
        if self.below_prob == 0:
            breakpoint()
        prob_cache[self] = self.below_prob
        #self.below_prob = self.below_prob
        return self.below_prob

    def propagate_above_probs(self,passed_above_prob): #reuse split_prob from propagate_below_probs
        """Recursively called on children, passing results from this node, so probs are propagated down the tree."""
        self.above_prob = passed_above_prob*self.split_prob
        for ps in self.splits:
            # this is how to get cousins probs
            ps['left'].propagate_above_probs(self.above_prob*ps['right'].below_prob)
            ps['right'].propagate_above_probs(self.above_prob*ps['left'].below_prob)
            assert np.allclose(ps['left'].above_prob*ps['left'].below_prob, ps['right'].above_prob*ps['right'].below_prob)
        if self.splits != []:
            if not self.above_prob > max([z.above_prob for ps in self.splits for z in (ps['right'],ps['left'])]):
                breakpoint()

    def info_if_leaf(self):
        shell_lf = self.lf.subtree_string(as_shell=True,alpha_normalized=True)
        lf = self.lf.subtree_string(alpha_normalized=True)
        word_str = ' '.join(self.words)
        #if shell_lf != alpha_normalize(shell_lf):
        #    breakpoint()
        #if lf != alpha_normalize(lf):
        #    breakpoint()
        return word_str, lf, shell_lf, self.sem_cats, self.sync

    def prob_as_leaf(self,syntaxl,shell_meaningl,meaningl,wordl,print_components=False):
        word_str, lf, shell_lf, sem_cats, sync = self.info_if_leaf()
        if sem_cats not in ({'Sq|NP','Sq|(S|NP)'},{'Sq|NP','Sq|(S|NP'}):
            for ssync in sem_cats:
                if not any(is_congruent(ssync,sc) for sc in self.sem_cats):
                    breakpoint()
                    raise SemCatError('Some syncat is not consistent with any semcat')
        #syntax_prob = max(syntaxl.prob('leaf',ssync) for ssync in syncs)
        syntax_prob = syntaxl.prob('leaf',sync)
        shmeaning_prob = max(shell_meaningl.prob(shell_lf, sc) for sc in sem_cats)
        meaning_prob = meaningl.prob(lf, shell_lf)
        word_prob = wordl.prob(word_str,lf)
        if print_components:
            for p in ('syntax_prob','shmeaning_prob','meaning_prob','word_prob'):
                exec(f'print("{p}:",round({p},5))')
        self.stored_prob_as_leaf = syntax_prob*shmeaning_prob*meaning_prob*word_prob
        if self.stored_prob_as_leaf==0:
            print('Zero prob for \n', self)
            raise ZeroProbError
        return self.stored_prob_as_leaf

    def gt_parse(self, depth=0):
        lfparts = re.split(r'[ \(\)\.]', self.lf_str)
        lfpartwords = [x.split('|')[1].lower().removesuffix('-bare').removesuffix('-prog') if '|' in x else x for x in lfparts]
        is_good = True
        texttree = '  '*depth + self.lf_str + ' ' + ' '.join(self.words) + f' {self.sync} prob: {self.below_prob:.03g} split prob: {self.split_prob:.03g}'
        for split in self.splits:
            lgtp, l_is_good, l_is_root_good = split['left'].gt_parse(depth+1)
            rgtp, r_is_good, r_is_root_good = split['right'].gt_parse(depth+1)
            if l_is_good and r_is_good:
                is_root_good = l_is_root_good and r_is_root_good
                return texttree + f'\n{lgtp}\n{rgtp}', True, is_root_good
        modified_forms_dict = {'ing':'prog', 'did':'do-past', 'wo':'will','n\'t':'not'}
        for w in self.words:
            if modified_forms_dict.get(w.lower(), w.lower()) in lfpartwords:
                continue
            elif w in ('is', '\'s', 'are', '\'re', 'were', 'am', '\'m', 'was'):
                if any(x in lfpartwords for x in ['equals', 'hasproperty', 'pres-1s', 'pres-2s', 'pres-3s', 'past-1s', 'past-2s', 'past-3s']):
                    continue
            elif w.endswith('ing') and w[:-3].lower() in lfpartwords:
                continue
            elif w.endswith('ing') and w[:-4].lower() in lfpartwords:
                continue
            elif w.endswith('ing') and w+'e' in lfpartwords:
                continue
            elif w.endswith('ing') and w[:-3]+'e' in lfpartwords:
                continue
            #elif w == 'ing':
            #    if 'prog' in lfpartwords:
            #        continue
            #elif w == 'did':
            #    if 'do-past' in lfpartwords:
            #        continue
            #elif w == 'wo':
            #    if 'will' in lfpartwords:
            #        continue
            is_good = False
        is_root_good = len(self.words)==1 and is_good
        return texttree, is_good, is_root_good

def can_compose_traised(fcat, gcat):
    f_cat_splits = split_respecting_brackets(fcat, sep='|')
    if len(f_cat_splits) == 2:
        return 'app'
    elif len(f_cat_splits) == 3 and f_cat_splits[1] == gcat:
        return 'cmp'
    return 'no'


def equal_maybe_typeraised(c1, c2):
    if c1=='X' or c2=='X':
        return True
    if c1==c2:
        return True
    if re.match(fr'([A-Z]{1,3})\|\(\1\|{c2}\)', c1): # c1 is c2 type raised
        return True
    if re.match(fr'([A-Z]{1,3})\|\(\1\|{c1}\)', c2): # c2 is c1 type raised
        return True
    return False
