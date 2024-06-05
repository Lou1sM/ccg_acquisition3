import numpy as np
from functools import partial
from utils import split_respecting_brackets, is_bracketed, all_sublists, maybe_brac, is_atomic, strip_string, cat_components, is_congruent, alpha_normalize, maybe_debrac, f_cmp_from_parent_and_g, combine_lfs, logical_type_raise, maybe_de_type_raise, logical_de_type_raise, is_wellformed_lf, is_type_raised, new_var_num, n_lambda_binders, set_congruent, lf_cat_congruent, is_cat_type_raised, lambda_match, is_bracket_balanced, apply_sem_cats, parent_cmp_from_f_and_g, balanced_substrings, non_directional
from errors import SemCatError, ZeroProbError, SynCatError
import re
import sys; sys.setrecursionlimit(500)
from learner_config import pos_marking_dict, base_lexicon, chiltag_to_node_types


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
    def __init__(self,defining_string,idx_in_tree=[],caches={'splits':{},'cats':{}},parent=None,dblfs=None,dbsss=None):
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
                self.extend_var_descs(int(self.string[1:]))
            elif self.string == 'and':
                self.node_type = 'connective'
                self.string = 'and'
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
                if chiltag not in chiltag_to_node_types:
                    if chiltag not in ['part','aux','on','poss','pro:exist','post']:
                        print(defining_string)
                    self.node_type = chiltag
                else:
                    self.node_type = chiltag_to_node_types[chiltag]
                #if '|' not in defining_string and not defining_string.startswith('$') and strip_string(defining_string) and sds!='':
            else:
                breakpoint()

        if self.is_leaf: # should only have children defined if comes from the __init__ in splits
            self.children = []

        if had_surrounding_brackets:
            assert self.subtree_string() == defining_string
        else:
            if not self.subtree_string() == defining_string:
                breakpoint()

        self.set_cats_from_string()

    def set_cats_from_string(self):
        if self.parent == 'START':
            self.is_semantic_leaf = False
        elif self.lf_str in self.caches['cats']:
            self.sem_cats, self.is_semantic_leaf, is_congruent = self.caches['cats'][self.lf_str]
            return is_congruent
        else:
            ss = self.stripped_subtree_string
            if debug_set_cats is not None and debug_set_cats==self.subtree_string(recompute=True):
                breakpoint()
            ss = ss.replace(' you','')
            if ss == 'not':
                self.is_semantic_leaf = True
                #self.sem_cats = set(['X'])
                self.sem_cats = set(['S|NP|(S|NP)'])
                had_initial_q = False
            elif ss == 'Q':
                self.is_semantic_leaf = True
                self.sem_cats = set(['X'])
                had_initial_q = True
            elif re.match(r'Q v\|do-(past|3s|2s|1s)_\d',ss):
                self.is_semantic_leaf = True
                self.sem_cats = set(['X'])
                had_initial_q = True
            else:
                had_initial_q = ss.startswith('Q ')
                if had_initial_q:
                    ss = ss.lstrip('Q ')
                    if not is_bracketed(ss):
                        breakpoint()
                    ss = ss[1:-1].strip()
                ss = maybe_debrac(ss[4:]) if (notstart:=ss.startswith('not ')) else ss
                self.is_semantic_leaf = ' ' not in ss.lstrip('BARE ')
                if not self.is_semantic_leaf:
                    self.sem_cats = set(['X'])
                elif ss == 'v|hasproperty':
                    self.is_semantic_leaf = True
                    self.sem_cats = set(['S|(N|N)|NP','S|NP|(N|N)'])
                elif '|' in ss:
                    pos_marking = ss.split('|')[0]
                    if pos_marking == 'n' and ss.rstrip('BARE').endswith('pl'):
                        self.sem_cats = set(['NP','N'])
                    elif pos_marking == 'n' and ss.endswith('BARE'):
                        self.sem_cats = set(['NP'])
                    elif pos_marking == 'n:prop' and ss.endswith('\'s\''):
                        self.sem_cats = set(['NP|N']) # e.g. John's
                    else:
                        self.sem_cats = pos_marking_dict.get(pos_marking,set(['X']))
                else:
                    word_level_form = ss.split('_')[0] if '_' in ss else ss
                    if word_level_form.startswith('Q '):
                        word_level_form = word_level_form[2:]
                    self.sem_cats = base_lexicon.get(word_level_form,set(['X']))
                if had_initial_q:
                    self.sem_cats = set('Sq'+sc[1:] if sc.startswith('S|') else sc for sc in self.sem_cats)
        assert self.sem_cats!=''
        assert self.sem_cats is not None
        assert not any(sc is None for sc in self.sem_cats)
        assert isinstance(self.sem_cats,set)
        is_congruent = self.is_type_congruent()
        if not (self.parent=='START'):
            self.caches['cats'][self.lf_str] = self.sem_cats, self.is_semantic_leaf, is_congruent
        return is_congruent

    def is_type_congruent(self):
        lf_str = logical_de_type_raise(self.lf_str) if self.is_type_raised() else self.lf_str
        if ' you' in lf_str and '.v|' in lf_str: # imperative
            if lf_str.startswith('lambda'): # transitive
                self.sem_cats = {'S|NP|NP'}
            else:
                self.sem_cats = {'S|NP'} # intransitive
        else:
            self.sem_cats = set(ssc for ssc in self.sem_cats if ssc=='X' or lf_cat_congruent(lf_str,ssc))
        return self.sem_cats != set([])

    def infer_splits(self):
        if self.is_semantic_leaf or self.n_lambda_binders > 4:
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
                #lmbdas_to_include = sorted(re.findall(r'\$\d{1,2}', mc))
                #lmbdas_to_include_ord_appear = sorted(lmbdas_to_include, key=lambda x:self.lf_str.index(x))
                lmbdas_to_include = sorted(re.findall(r'\$\d{1,2}', mc), key=lambda x:self.lf_str.index(x))
                assert all('lambda '+x in self.lf_str.rpartition('.')[0] for x in lmbdas_to_include)
                g_lambda_terms = '.'.join(f'lambda {x}' for x in lmbdas_to_include)
                gcmp = LogicalForm(f'{g_lambda_terms}.{maybe_debrac(mc)}', caches=self.caches)
                if all(is_atomic(gsc) for gsc in gcmp.sem_cats): # this may be blocking traising
                    continue
                if gcmp.sem_cats == {'NP|N'}: # don't compose with determiners
                    continue
                fghole = lmbdas_to_include[0] if len(lmbdas_to_include)==1 else '('+' '.join(lmbdas_to_include)+')'
                f_cmp_string = self.lf_str.replace(mc, fghole)
                if strip_string(f_cmp_string) == 'Q':
                    continue # no just Q nodes for now
                fcmp = LogicalForm(f_cmp_string, caches=self.caches)
                if not fcmp.set_cats_from_string():
                    continue
                if (gcmp.n_lambda_binders > 4) or (gcmp.stripped_subtree_string)=='': # don't consider only variables
                    breakpoint()
                    continue
                self.add_split(fcmp, gcmp, 'cmp')
        if has_q:=any(x.node_type=='Q' for x in self.descs):
            qidx = [i for i,x in enumerate(self.descs) if x.node_type=='Q'][0]
            head_options = [x for x in self.leaf_descs if x.node_type in ['verb','raise','quant']]
            #head_options = [x for x in self.leaf_descs if x.node_type in ['quant','raise']]
            if len(head_options)==0:
                head_options = [x for x in self.leaf_descs if x.node_type in ['quant','prep','adv','aux']]
            if len(head_options)==0:
                self.app_splits = set()
                self.cmp_splits = set()
                return self.app_splits
            else:
                head = head_options[0]

        #remove_types = ['entity','verb','quant','noun','neg','raise','WH','adj','prep']
        remove_types = ['entity','verb','quant','neg','raise','WH','adj','prep','noun','hasproperty'] # no nested nouns
        pridxs = [i for i,d in enumerate(self.descs) if d.node_type in remove_types]
        #if self.n_lambda_binders==2 and len(pridxs)==2:
        #    if re.match(fr'^lambda \$(\d{{1,2}})\.lambda \$(\d{{1,2}})\.{re.escape(self.descs[pridxs[0]].lf_str)} \({re.escape(self.descs[pridxs[1]].lf_str)} \(\$\1 \$\2\)\)', self.lf_str):
        #        #f_cmp = LogicalForm(f'lambda ${vnum+1}.lambda ${vnum+2}. {self.descs[pridxs[0]]} (${vnum+1} ${vnum+2}')
        #        f_cmp = LogicalForm(f'lambda $0.lambda $1. {self.descs[pridxs[0]].lf_str} ($0 $1)', caches=self.caches)
        #        g_cmp = LogicalForm(f'lambda $0.lambda $1. {self.descs[pridxs[1]].lf_str} ($0 $1)', caches=self.caches)
        #        self.add_split(f_cmp, g_cmp, 'cmp')
        #    elif re.match(fr'^lambda \$(\d{{1,2}})\.lambda \$(\d{{1,2}})\.{re.escape(self.descs[pridxs[1]].lf_str)} \({re.escape(self.descs[pridxs[0]].lf_str)} \(\$\1 \$\2\)\)', self.lf_str):
        #        #f_cmp = LogicalForm(f'lambda ${vnum+1}.lambda ${vnum+2}. {self.descs[pridxs[0]]} (${vnum+1} ${vnum+2}')
        #        f_cmp = LogicalForm(f'lambda $0.lambda $1. {self.descs[pridxs[1]].lf_str} ($0 $1)', caches=self.caches)
        #        g_cmp = LogicalForm(f'lambda $0.lambda $1. {self.descs[pridxs[0]].lf_str} ($0 $1)', caches=self.caches)
        #        self.add_split(f_cmp, g_cmp, 'cmp')
        for ridxs in all_sublists(pridxs):
            if (n_removees := len(ridxs)) in (0,len(pridxs)): continue
            if self.sem_cats.intersection({'S','S|NP'}) and \
                all(self.descs[ri].sem_cats.intersection({'N'}) for ri in ridxs):
                    continue # this would result in illegitimate cats, like S|N or S|NP|N
            not_removees = [d for i,d in enumerate(self.descs) if i not in ridxs]
            if any(self.descs[i].lf_str==d.lf_str for i in ridxs for d in not_removees):
                continue # all or nothing rule
            f = self.copy()
            f.parent = self
            to_remove = [f.descs[i] for i in ridxs]
            if has_q and head in to_remove:
                ridxs.append(qidx)
            if any(x in not_removees for x in to_remove):
                breakpoint()
                continue
            if any([x.node_type=='detnoun' for x in to_remove]):
                breakpoint()
            assert all([n.node_type in remove_types+['noun'] for n in to_remove])
            leftmost = f.descs[min(ridxs)]
            entry_point = leftmost # where to substitute g into f
            changed = True
            # find the lowest entry point that has all to_removes as descs
            while changed:
                if entry_point.parent is None:
                    break
                changed = False
                for tr in to_remove:
                    if tr not in entry_point.descs:
                        changed = True
                        entry_point = entry_point.parent
                        break
            if entry_point.sem_cats in [{'S|NP'}, {'S|NP|NP'}] and len(entry_point.parent.children) == len(entry_point.parent.var_descs)+1:
                breakpoint()
                #print('stepping up to get vars')
                #entry_point = entry_point.parent
            g = entry_point.copy()
            to_present_as_args_to_g = '' if len(to_remove)==1 and to_remove[0].node_type=='detnoun' else [d for d in entry_point.leaf_descs if d not in to_remove]
            have_embedded_binders = [d for d in to_present_as_args_to_g if d.node_type=='bound_var' and d.binder in entry_point.descs]
            to_present_as_args_to_g = [x for x in to_present_as_args_to_g if x not in have_embedded_binders]
            assert len(to_present_as_args_to_g) == len(entry_point.leaf_descs) - n_removees - len(have_embedded_binders)
            g = g.turn_nodes_to_vars(to_present_as_args_to_g)
            if (g.n_lambda_binders > 4) or (strip_string(g.subtree_string().replace(' AND','')) == ''): # don't consider only variables
                continue
            g_sub_var_num = self.new_var_num
            new_entry_point_in_f_as_str = ' '.join([f'${g_sub_var_num}'] + list(reversed([maybe_brac(n.string,sep=' ') for n in to_present_as_args_to_g])))
            assert entry_point.subtree_string() in self.subtree_string()
            if any(x is y for x in self.descs for y in f.descs):
                breakpoint()
            entry_point.__init__(new_entry_point_in_f_as_str,self.idx_in_tree,entry_point.caches)
            if len(to_present_as_args_to_g) >= 3: # then f will end up with arity 4, too high
                continue
            f = f.lambda_abstract(g_sub_var_num)
            if not f.set_cats_from_string():
                continue
            if 'mod|' in f.lf_str and g.is_leaf and g.lf_str.startswith('v|'):
                f.infer_splits()
                if any(sc in f.sem_cats for sc in ['S|NP|(S|NP)', 'S|(S|NP)', 'Sq|(S|NP)']):
                    g = g.spawn_self_like(f'lambda $0.{g.subtree_string()} $0')
            self.add_split(f,g,'app')
            # if self is a lambda and had it's bound var removed and put in g then try cmp
            if self.node_type == 'lmbda' and any(not is_atomic(x) for x in self.sem_cats) and any(not is_atomic(x) for x in f.sem_cats) and f.node_type == 'lmbda' and any(cat_components(x,allow_atomic=True)[-1] == cat_components(y,allow_atomic=True)[-1] and cat_components(y,allow_atomic=True)[-1]!='X' for x in self.sem_cats for y in f.sem_cats):
                f_cmp_string = logical_type_raise(g.lf_str)
                assert f_cmp_string == logical_type_raise(g.lf_str)
                f_cmp = LogicalForm(f_cmp_string, caches=self.caches)
                fparts = split_respecting_brackets(f.subtree_string(),sep='.')
                # g_cmp is just g reordered, move the part at n_removees position to front
                # because first var will get chopped off from cmp and need
                # remainder to be able to receive body of f_cmp just like f received g
                g_cmp_parts = fparts[1:2] + fparts[:1] + fparts[2:]
                g_cmp_string = '.'.join(g_cmp_parts)
                g_cmp = LogicalForm(g_cmp_string, caches=self.caches)
                if not (g_cmp.sem_cat_is_set and all(is_atomic(gcsc) for gcsc in g_cmp.sem_cats)): # g_cmp must have slash
                    self.add_split(f_cmp,g_cmp,'cmp')
        if '|' in self.sem_cats:
            assert self.subtree_string().startswith('lambda')
        if self.sem_cats == set():
            raise SemCatError('empty semcats after inferring splits')
        self.put_self_in_caches()
        return self.app_splits, self.cmp_splits

    def add_split(self,f,g,split_type):
        f.parent = g.parent = self
        if not g.set_cats_from_string():
            return
        if g.sem_cats == {'NP|N'} and not g.lf_str.split('.')[1].startswith('not') and not g.lf_str.split('.')[1].startswith('Q'):
            breakpoint()
        if not g.cat_consistent():
            return
        assert f != self
        assert g != self
        if split_type == 'cmp':
            assert f.subtree_string().startswith('lambda')
            assert g.subtree_string().startswith('lambda')
            #f.sem_cats = set(['X'])
            to_add_to = self.cmp_splits
        assert ( combine_lfs(f.subtree_string(),g.subtree_string(),split_type,normalize=True) == self.subtree_string(alpha_normalized=True))
        g.infer_splits()
        if not ('mod|' in f.lf_str and g.is_leaf and g.lf_str.startswith('v|')):
            f.infer_splits() # don't do again if already done in infer_splits
        if self.sem_cat_is_set and g.sem_cat_is_set:
            self.set_f_sem_cat_from_self_and_g(f,g,split_type)
            if not f.cat_consistent():
                return
        if f.sem_cats.intersection(['S|N','S|NP|N']):
            breakpoint()
            return
        comb_fn = apply_sem_cats if split_type=='app' else partial(parent_cmp_from_f_and_g, sem_only=True)
        #if self.lf_str == 'lambda $1.v|eat $1 n|tiger-pl-BARE':
            #breakpoint()
        fcats_to_use = f.sem_cats if split_type=='app' else set(c for c in f.sem_cats if '|' in c)
        gcats_to_use = g.sem_cats if split_type=='app' else set(c for c in g.sem_cats if '|' in c)
        if g.sem_cat_is_set and f.sem_cat_is_set: # no cmp inferred cats ftm
            inferred_sem_cats = set([])
            for fsc in fcats_to_use:
                for gsc in gcats_to_use:
                    new_inferred_sem_cat = comb_fn(fsc, gsc)
                    if new_inferred_sem_cat is not None:
                        self.sem_cats = self.sem_cats | set([new_inferred_sem_cat])
        if self.sem_cats != {'X'}:
            self.sem_cats = set(sc for sc in self.sem_cats if sc!='X')
        #if inferred_sem_cats:
            #if self.sem_cats == {'X'}:
                #self.sem_cats = inferred_sem_cats
            #else:
                #new = self.sem_cats.intersection(inferred_sem_cats)
                #if not new:
                    #breakpoint()
                #self.sem_cats = new
            if not self.cat_consistent():
                return
            g.sem_cats = set([gsc for gsc in gcats_to_use if 'X' in gsc or any(comb_fn(fsc, gsc)==sc for fsc in fcats_to_use for sc in self.sem_cats)])
            f.sem_cats = set([fsc for fsc in fcats_to_use if 'X' in fsc or any(comb_fn(fsc, gsc)==sc for gsc in gcats_to_use for sc in self.sem_cats)])
        if not ( ( f.sem_cat_is_set or not self.sem_cat_is_set or not g.sem_cat_is_set)):
            breakpoint()
        if not f.sem_cat_is_set:
            #print(f.lf_str,' + ',g.lf_str)
            return
        if not ( not (self.sem_cats == set(['S']) and g.sem_cats == set(['N']))):
            breakpoint()
        f.sibling = g
        g.sibling = f
        if f.is_type_congruent():
            f.put_self_in_caches()
            g.put_self_in_caches()
            if split_type=='app':
                self.app_splits = self.app_splits | {(f,g)}
            else:
                assert split_type == 'cmp'
                self.cmp_splits = self.cmp_splits | {(f,g)}
        assert self.sem_cats != set()

    def put_self_in_caches(self):
        existing_app_splits, existin_cmp_splits, existing_sem_cats = self.caches['splits'].get(self.lf_str, ({},{},{}))
        self.caches['splits'][self.lf_str] = self.app_splits.union(existing_app_splits), self.cmp_splits.union(existin_cmp_splits), self.sem_cats.union(existing_sem_cats)

    def cat_consistent(self):
        """Return False if sem_cat says type-raised but lf not of type-raised form."""
        if self.sem_cats.intersection(['S|N','Sq|N']):
            return False
        if 'mod|' in self.lf_str: # mod messes it up because looks type-raised
            return True
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
        if new_inferred_sem_cats.intersection(set(['S|N','S|NP|N'])):
            return
        old_fsc = f.sem_cats
        if f.sem_cats | new_inferred_sem_cats in [{'Sq|NP','Sq|(S|NP)'},{'S|NP','S|(S|NP)'}]:
            print(self, f, g)
            f.sem_cats = f.sem_cats | new_inferred_sem_cats
        else:
            #f.sem_cats = f.sem_cats.intersection(new_inferred_sem_cats) if f.sem_cat_is_set else new_inferred_sem_cats
            f.sem_cats = set_congruent(new_inferred_sem_cats, f.sem_cats)
        if not f.sem_cats:
            raise SemCatError('no f-semcats are congruent with it\'s new inferred semcats')
            #print(f'fsem_cats were {old_fsc}, inferred are {new_inferred_sem_cats}')
        if f.sem_cats == {'Sq|(S|(S|NP))'}:
            breakpoint()
        for f1,g1 in f.app_splits:
            if g1.sem_cat_is_set and not (all(is_atomic(gsc) for gsc in g.sem_cats) and comb_type=='cmp'):
                f.set_f_sem_cat_from_self_and_g(f1,g1,comb_type)

    def spawn_child(self,defining_string,sibling_idx):
        """Careful, this means a child in the tree of one logical form, not a possible split."""
        return LogicalForm(defining_string,idx_in_tree=self.idx_in_tree+[sibling_idx],caches=self.caches,parent=self)

    def spawn_self_like(self,defining_string,idx_in_tree=None):
        if idx_in_tree is None:
            idx_in_tree = self.idx_in_tree
        new = LogicalForm(defining_string,idx_in_tree=idx_in_tree,caches=self.caches,parent=self.parent)
        new.sem_cats = self.sem_cats
        return new

    def is_type_raised(self):
        is_tr = any(is_cat_type_raised(ssc) for ssc in self.sem_cats)
        assert is_tr == any(is_cat_type_raised(ssc) for ssc in self.sem_cats)
        return is_tr

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
            #bracced_if_var = f'({x})' if x.startswith('$') or x.startswith('BARE') else x
            #return f'Q {bracced_if_var}'
            return f'Q {maybe_brac(x)}'
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
        to_remove = [d for d in copied.descs if d in nodes]
        # node may be 'in' another list if there is another node in that list
        # that has the same defining string as node, but we want the reference,
        # i.e. node itself, rather than its copy in the list
        vars_abstracted = []
        for desc in to_remove:
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
    def __init__(self,lf,words,node_type,parent=None,sibling=None,syncs=None):
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
        if len(self.lf.app_splits) == 0:
            self.lf.infer_splits()
        self.sem_cats = self.lf.sem_cats
        if syncs is not None:
            self.syncs = syncs
        else:
            self.syncs = lf.sem_cats
        self.is_leaf = self.lf.is_semantic_leaf or len(self.words) == 1
        if not self.is_leaf:
            for split_point in range(1,len(self.words)):
                left_words = self.words[:split_point]
                right_words = self.words[split_point:]
                for f,g in self.lf.app_splits:
                    if g.sem_cat_is_set:
                        self.add_app_splits(f,g,left_words,right_words)
                for f,g in self.lf.cmp_splits:
                    if g.sem_cat_is_set:
                        self.add_cmp_splits(f,g,left_words,right_words,'fwd')
                        self.add_cmp_splits(f,g,left_words,right_words,'bck')
                    if 'P' in self.syncs:
                        breakpoint()

        if len(self.splits) == 0:
            self.is_leaf = True
        assert not any(x is None for x in self.sem_cats)
        assert not any(x is None for x in self.syncs)
        if len(self.syncs)==0:
            raise SynCatError('parse node is left with no suitable syncats')

    def add_cmp_splits(self, f, g, left_words, right_words, direction):
        # just fwd_cmp for now
        if direction=='fwd':
            g_child = ParseNode(g,right_words,parent=self,node_type='right_fwd_cmp')
        else:
            assert direction=='bck'
            g_child = ParseNode(g,left_words,parent=self,node_type='right_bck_cmp')
        inferred_f_syncs = set([])
        inferred_g_syncs = set([])
        for ssync in self.syncs:
            for gsync in g_child.syncs:
                new_inferred_f_sync, new_inferred_g_sync = f_cmp_from_parent_and_g(ssync,gsync,sem_only=False)
                #if not ( is_congruent(gsync, new_inferred_g_sync)) and self.parent.lf_str!='lambda $0.not (mod|will (v|eat pro:per|you $0))':
                if is_congruent(gsync, new_inferred_g_sync):
                    if new_inferred_f_sync: inferred_f_syncs.add(new_inferred_f_sync)
                    if new_inferred_g_sync: inferred_g_syncs.add(new_inferred_g_sync)
                elif not (gsync=='S|NP|(S|NP)' and non_directional(ssync)=='S|(S|NP)|NP'):
                    print(f'excluding gsync {gsync} and ssync {ssync} from {self.lf_str} {self.syncs}')

        assert 'S/NP\\NP' not in inferred_f_syncs
        if None in inferred_f_syncs:
            breakpoint()
        if inferred_f_syncs == set([]) or inferred_f_syncs == {None}:
            return
        if direction=='fwd':
            f_child = ParseNode(f,left_words,parent=self,node_type='left_fwd_cmp',syncs=inferred_f_syncs)
        else:
            f_child = ParseNode(f,right_words,parent=self,node_type='left_bck_cmp',syncs=inferred_f_syncs)
        if None in inferred_f_syncs:
            breakpoint()
        g_child.syncs = set([s for s in g_child.syncs if any(is_congruent(s,y) for y in inferred_f_syncs)])
        if None in f.sem_cats:
            breakpoint()
        assert set_congruent(f.sem_cats,inferred_f_syncs)
        assert set_congruent(g.sem_cats,inferred_g_syncs)
        g_child.syncs = inferred_g_syncs
        if direction=='fwd':
            self.append_split(f_child,g_child,f'{direction}_cmp')
        else:
            self.append_split(g_child,f_child,f'{direction}_cmp')

    def add_app_splits(self,f,g,left_words,right_words):
        # CONVENTION: f-child with the words on the left of sentence is the
        # 'left' child, even if bck application, left child is g in that case
        right_child_fwd = ParseNode(g,right_words,parent=self,node_type='right_fwd_app')
        new_syncs = set(f'{ssync}/{maybe_brac(rcsync)}' for ssync in self.syncs for rcsync in right_child_fwd.syncs)
        assert new_syncs != 'S/NP\\NP'
        bad_new_syncs = [sc for sc in ('X\\S', 'X/S') if sc in new_syncs]
        if len(bad_new_syncs) > 0:
            raise SemCatError(f"bad new syn cats: {''.join(bad_new_syncs)}")
        left_child_fwd = ParseNode(f,left_words,parent=self,node_type='left_fwd_app',syncs=new_syncs)
        #congs = set(x for x in f.sem_cats if any(is_congruent(x,y) for y in new_syncs))
        if not (f.was_cached or self.syncs==set('X') or set_congruent(f.sem_cats,new_syncs)):
            raise SemCatError('no f-semcats are congruent with it\'s new inferred syncats')
        self.append_split(left_child_fwd,right_child_fwd,'fwd_app')

        left_child_bck = ParseNode(g,left_words,parent=self,node_type='left_bck_app')
        new_syncs = set(f'{ssync}\\{maybe_brac(lcsync)}' for ssync in self.syncs for lcsync in left_child_bck.syncs if not (bool(re.match(r'S/[A-Za-z]+$',ssync)) and lcsync=='NP'))
        if new_syncs and self.syncs!=set('X'):
            assert f.was_cached or set_congruent(f.sem_cats, new_syncs)
            try:
                right_child_bck = ParseNode(f,right_words,parent=self,node_type='right_bck_app',syncs=new_syncs)
                self.append_split(left_child_bck,right_child_bck,'bck_app')
            except SynCatError:
                print(f'Syncat error when making node {f} {right_words}')

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

    def info_if_leaf(self):
        shell_lf = self.lf.subtree_string(as_shell=True,alpha_normalized=True)
        lf = self.lf.subtree_string(alpha_normalized=True)
        word_str = ' '.join(self.words)
        sem_cats = set(maybe_debrac(maybe_de_type_raise(ssc)) for ssc in self.sem_cats)
        syncs = set(maybe_debrac(maybe_de_type_raise(ssc)) for ssc in self.syncs)
        if sem_cats != self.sem_cats: # has been de-type-raised
            assert syncs != self.syncs or any('X' in sync for sync in syncs)
            assert all(any(maybe_debrac(maybe_de_type_raise(s1))==s2 for s1 in self.sem_cats) for s2 in sem_cats)
            assert all(any(maybe_debrac(maybe_de_type_raise(s1))==s2 for s1 in self.syncs) for s2 in syncs)
            shell_lf = alpha_normalize(logical_de_type_raise(shell_lf))
            lf = alpha_normalize(logical_de_type_raise(lf))
        return word_str, lf, shell_lf, sem_cats, syncs

    def __repr__(self):
        base = (f"ParseNode\n"
                f"\tWords: {' '.join(self.words)}\n"
                f"\tLogical Form: {self.lf.subtree_string()}\n"
                f"\tSyntactic Category: {self.syncs}\n")
        if hasattr(self,'stored_prob'):
            base += f'\tProb: {self.stored_prob}\n'
        return base

    def show_splits(self):
        for i,split in enumerate(self.splits):
            f, g = split['left'], split['right']
            fwords, gwords = ' '.join(f.words), ' '.join(g.words)
            print(f'{i}: {f.lf_str} {fwords}\t+\t{g.lf_str} {gwords}')

    def __eq__(self,other):
        if not isinstance(other,LogicalForm):
            return False
        return self.lf_str == other.lf_str and self.words == other.words

    def __hash__(self):
        return hash(self.lf_str + ' '.join(self.words))

    def siblingify(self,other):
        self.sibling = other
        other.sibling = self
#

    def propagate_below_probs(self,syntaxl,shell_meaningl,meaningl,wordl,prob_cache,split_prob,is_map):
        self.split_prob = split_prob
        if self in prob_cache:
            return prob_cache[self]
        all_probs = [self.prob_as_leaf(syntaxl,shell_meaningl,meaningl,wordl)]
        for ps in self.splits:
            split_prob = max(syntaxl.prob(f'{psl} + {psr}',ss) for psl in ps['left'].syncs for psr in ps['right'].syncs for ss in self.syncs)
            if split_prob == 0:
                breakpoint()
            left_below_prob = ps['left'].propagate_below_probs(syntaxl,shell_meaningl,meaningl,wordl,prob_cache,split_prob,is_map)
            right_below_prob = ps['right'].propagate_below_probs(syntaxl,shell_meaningl,meaningl,wordl,prob_cache,split_prob,is_map)
            all_probs.append(left_below_prob*right_below_prob*split_prob)

        below_prob = max(all_probs) if is_map else sum(all_probs)
        if below_prob == 0:
            breakpoint()
        prob_cache[self] = below_prob
        self.below_prob = below_prob
        return below_prob

    def propagate_above_probs(self,passed_above_prob): #reuse split_prob from propagate_below_probs
        self.above_prob = passed_above_prob*self.split_prob
        for ps in self.splits:
            # this is how to get cousins probs
            ps['left'].propagate_above_probs(self.above_prob*ps['right'].below_prob)
            ps['right'].propagate_above_probs(self.above_prob*ps['left'].below_prob)
            assert np.allclose(ps['left'].above_prob*ps['left'].below_prob, ps['right'].above_prob*ps['right'].below_prob)
        if self.splits != []:
            if not self.above_prob > max([z.above_prob for ps in self.splits for z in (ps['right'],ps['left'])]):
                breakpoint()

    def prob_as_leaf(self,syntaxl,shell_meaningl,meaningl,wordl,print_components=False):
        word_str, lf, shell_lf, sem_cats, syncs = self.info_if_leaf()
        if sem_cats not in ({'Sq|NP','Sq|(S|NP)'},{'Sq|NP','Sq|(S|NP'}):
            for sc in sem_cats:
                if not any(is_congruent(ssync,sc) for ssync in syncs):
                    raise SemCatError('Some syncat is not consistent with any semcat')
        syntax_prob = max(syntaxl.prob('leaf',ssync) for ssync in syncs)
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
