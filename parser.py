from utils import split_respecting_brackets, is_bracketed, all_sublists, maybe_brac, is_atomic, strip_string, concat_lfs, cat_components, is_congruent, first_lambda_body_split, alpha_normalize, maybe_debrac, parent_cmp_from_f_and_g, f_cmp_from_parent_and_g, num_nps, combine_lfs, logical_type_raise, maybe_de_type_raise, logical_de_type_raise, is_wellformed_lf, is_type_raised, new_var_num
import re


# is_leaf means it's atomic in lambda calculus
# is_semantic_leaf means we shouldn't consider breaking it further
# e.g. lambda $0. runs $0 is a semantic leaf but not a leaf

# split_prob is the prob, according to the syntax_learner, of the split that
# gave birth to it; above_prob is the prob of the branch this node is on, equal
# to split_prob*parent.above_prob*sibling.below_prob; below_prob is the prob of
# this node, under all possible ways of splitting it down to the leaves;

# Q(.) is like a composite, or composite is like a dummy wrapper around an lf,
# Q then being an alternative non-dummy wrapper

class LogicalForm:
    def __init__(self,defining_string,base_lexicon,idx_in_tree=[],caches=None,parent=None):
        assert isinstance(idx_in_tree,list)
        assert is_wellformed_lf(defining_string)
        had_surrounding_brackets = False
        self.base_lexicon = base_lexicon
        self.idx_in_tree = idx_in_tree
        self.caches = {'splits':{},'sem_cats':{},'lfs':{}} if caches is None else caches
        self.var_descendents = []
        self.possible_app_splits = []
        self.possible_cmp_splits = []
        self.parent = parent
        self.stored_subtree_string = ''
        self.stored_shell_subtree_string = ''
        self.stored_alpha_normalized_subtree_string = ''
        self.stored_alpha_normalized_shell_subtree_string = ''
        defining_string = maybe_debrac(defining_string)
        if parent == 'START':
            if defining_string.startswith('Q'):
                self.sem_cat = 'Sq'
            else:
                self.sem_cat = 'S'

        if bool(re.match(r'^lambda \$\d',defining_string)):
            ss = re.search(r'(?<=lambda \$\d\.)([a-z])*(?= \$\d{1,2}$)',defining_string)
            if ss is not None and self.base_lexicon[ss.group()] == 'N':
                self.children = []
                self.is_leaf = True
                self.node_type = 'noun'
                self.string = defining_string
            else:
                lambda_string, _, remaining_string = defining_string.partition('.')
                variable_index = lambda_string[-2:]
                self.is_leaf = False
                self.string = lambda_string
                self.node_type = 'lmbda'
                self.children = [self.spawn_child(remaining_string,0)]

                for d in self.descendents:
                    if d.node_type == 'unbound_var' and d.string == variable_index:
                        d.binder = self
                        d.node_type = 'bound_var'
        elif bool(re.match(r'^Q\(.*\)$',defining_string)):
            self.node_type = 'Q'
            self.string = 'Q'
            #self.children = [self.spawn_child(cstr,i) for i,cstr in enumerate(split_respecting_brackets(defining_string[2:-1]))]
            self.children = [self.spawn_child(defining_string[2:-1],0)]
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
                self.node_type = 'dconst' # determiner + noun treated kinda like a const
            else:
                self.node_type = 'composite'
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
            elif self.string in ['a','the']:
                self.node_type = 'quant'
            else:
                self.node_type = 'const'
        if had_surrounding_brackets:
            if not self.subtree_string() == defining_string[1:-1]:
                breakpoint()
        else:
            if not self.subtree_string() == defining_string:
                breakpoint()

        self.is_semantic_leaf = self.is_leaf # is_leaf is sufficient for is_semantic_leaf
        self.set_cats_from_string()

    def set_cats_from_string(self):
        if self.parent != 'START':
            ss = self.stripped_subtree_string
            if ss in self.base_lexicon:
                self.sem_cat = self.base_lexicon[ss]
                self.is_semantic_leaf = True # but not necessary
            elif self.node_type == 'dconst':
                self.sem_cat = 'NP'
            else:
                self.sem_cat = 'X'
        assert self.sem_cat!=''

    def infer_splits(self):
        if self.is_semantic_leaf or self.num_lambda_binders > 4:
            self.possible_app_splits = []
            return self.possible_app_splits
        if self.lf_str in self.caches['splits']:
            self.possible_app_splits = self.caches['splits'][self.lf_str]
            return self.possible_app_splits
        possible_removee_idxs = [i for i,d in enumerate(self.descendents) if d.node_type in ['const','noun','quant']]
        for removee_idxs in all_sublists(possible_removee_idxs):
            if self.lf_str == 'Q(buy chicago montgomery)' and removee_idxs == [2,4]:
                breakpoint()
            n_removees = len(removee_idxs)
            if n_removees == 0: continue
            if n_removees == len(possible_removee_idxs) and self.node_type!='Q': continue
            f = self.copy()
            f.parent = self
            to_remove = [f.descendents[i] for i in removee_idxs]
            assert all([n.node_type in ['const','dconst','noun','quant'] for n in to_remove])
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
            if entry_point.node_type == 'Q':
                g = self.spawn_self_like(' '.join([maybe_brac(c.lf_str,sep=' ')
                    for c in entry_point.children]),idx_in_tree=entry_point.idx_in_tree)
                assert all([c1.lf_str==c2.lf_str for c1,c2 in zip(g.descendents,entry_point.descendents)][1:])
                assert entry_point.descendents[0].lf_str != g.descendents[0].lf_str
            else:
                g = entry_point.copy()
            to_present_as_args_to_g = '' if len(to_remove)==1 and to_remove[0].node_type=='dconst' else [d for d in entry_point.leaf_descendents if d not in to_remove]
            assert len(to_present_as_args_to_g) == len(entry_point.leaf_descendents) - n_removees
            g = g.turn_nodes_to_vars(to_present_as_args_to_g)
            if (g.num_lambda_binders > 4) or (strip_string(g.subtree_string().replace(' AND','')) == ''): # don't consider only variables
                continue
            g_sub_var_num = self.new_var_num
            new_entry_point_in_f_as_str = ' '.join([f'${g_sub_var_num}'] + list(reversed([maybe_brac(n.string,sep=' ') for n in to_present_as_args_to_g])))
            assert entry_point.subtree_string() in self.subtree_string()
            entry_point.__init__(new_entry_point_in_f_as_str,entry_point.base_lexicon,self.idx_in_tree,entry_point.caches)
            if len(to_present_as_args_to_g) >= 3: # then f will end up with arity 4
                continue
            if strip_string(f.subtree_string().replace(' AND','')) == '': # exclude just 'AND's
                continue
            f.set_cats_from_string()
            f = f.lambda_abstract(g_sub_var_num)
            self.add_split(f,g,'app')
            # if self is a lambda and had it's bound var removed and put in g then try cmp
            if self.node_type == 'lmbda' and f.node_type == 'lmbda' and cat_components(self.sem_cat,allow_atomic=True)[-1] == cat_components(f.sem_cat,allow_atomic=True)[-1] and cat_components(f.sem_cat,allow_atomic=True)[-1]!='X':
                #assert g.num_lambda_binders > n_removees
                #f_cmp_string = f"lambda ${g.new_var_num}.{lambda_binder}${g.new_var_num} {maybe_brac(rest,sep=' ')}"
                f_cmp_string = logical_type_raise(g.lf_str)
                assert f_cmp_string == logical_type_raise(g.lf_str)
                f_cmp = LogicalForm(f_cmp_string,self.base_lexicon)
                fparts = split_respecting_brackets(f.subtree_string(),sep='.')
                # g_cmp is just g reordered, move the part at n_removees position to front
                # because first var will get chopped off from cmp and need
                # remainder to be able to receive body of f_cmp just like f received g
                #assert bool(re.match(r'lambda \$\d{1,2}',fparts[n_removees]))
                g_cmp_parts = [fparts[f.num_lambda_binders-1]] + fparts[:f.num_lambda_binders-1] + fparts[f.num_lambda_binders:]
                g_cmp_string = '.'.join(g_cmp_parts)
                g_cmp = LogicalForm(g_cmp_string,self.base_lexicon)
                self.add_split(f_cmp,g_cmp,'cmp')
        if '|' in self.sem_cat:
            assert self.subtree_string().startswith('lambda')
        self.caches['splits'][self.lf_str] = self.possible_app_splits
        return self.possible_app_splits

    def add_split(self,f,g,split_type):
        f.parent = g.parent = self
        g.set_cats_from_string()
        if g.sem_cat not in ['X','N'] and len(re.findall(r'[\\/\|]',g.sem_cat)) != g.num_lambda_binders:
            return
        if not g.type_raised_consistent():
            return
        assert f != self
        assert f.sem_cat != ''
        assert g != self
        assert g.sem_cat != ''
        if split_type == 'app':
            f.set_cats_from_string()
            if f.sem_cat not in ['N','X'] and f.num_lambda_binders not in [0,-num_nps(f.sem_cat)]:
                return
            to_add_to = self.possible_app_splits
        else:
            assert split_type == 'cmp'
            assert f.subtree_string().startswith('lambda')
            assert g.subtree_string().startswith('lambda')
            f.sem_cat = 'X'
            assert not (g.sem_cat_is_set and is_atomic(g.sem_cat))
            to_add_to = self.possible_cmp_splits
        if not combine_lfs(f.subtree_string(),g.subtree_string(),split_type,normalize=True)== self.subtree_string(alpha_normalized=True):
            breakpoint()
        g.infer_splits()
        if self.sem_cat_is_set and g.sem_cat_is_set:
            self.propagate_sem_cat_leftward(f,g,split_type)
            if not f.type_raised_consistent():
                return
        f.infer_splits()
        if g.sem_cat_is_set:
            if f.sem_cat_is_set:
                #assert split_type == 'app' don't know why I had this in
                hits = re.findall(fr'[\\/\|]\(?{re.escape(g.sem_cat)}\)?$',re.escape(f.sem_cat))
                assert len(hits) < 2
                for hit in hits:
                    new_inferred_sem_cat = f.sem_cat[:-len(hit)]
                    assert self.sem_cat in ['X',new_inferred_sem_cat]
                    self.sem_cat = new_inferred_sem_cat
                    if not self.type_raised_consistent():
                        return
        assert f.sem_cat_is_set or not self.sem_cat_is_set or not g.sem_cat_is_set
        if (f,g) in to_add_to:
            breakpoint()
        to_add_to.append((f,g))
        if '/' in f.sem_cat:
            breakpoint()

    def type_raised_consistent(self):
        """Return False if sem_cat says type-raised but lf not of type-raised form."""
        return self.sem_cat !='S|(S|NP)' or is_type_raised(self.subtree_string())

    def propagate_sem_cat_leftward(self,f,g,comb_type):
        if comb_type == 'app':
            new_inferred_sem_cat = f'{self.sem_cat}|{maybe_brac(g.sem_cat)}'
        else:
            assert comb_type == 'cmp'
            new_inferred_sem_cat,_ = f_cmp_from_parent_and_g(self.sem_cat,g.sem_cat,sem_only=True)
        if new_inferred_sem_cat.startswith('|'):
            breakpoint()
        assert is_congruent(f.sem_cat,new_inferred_sem_cat)
        f.sem_cat = new_inferred_sem_cat
        for f1,g1 in f.possible_app_splits:
            if g1.sem_cat_is_set and not (is_atomic(g1.sem_cat) and comb_type=='cmp'):
                f.propagate_sem_cat_leftward(f1,g1,comb_type)

    def spawn_child(self,defining_string,sibling_idx):
        """Careful, this means a child in the tree of one logical form, not a possible split."""
        return LogicalForm(defining_string,idx_in_tree=self.idx_in_tree+[sibling_idx],base_lexicon=self.base_lexicon,caches=self.caches,parent=self)

    def spawn_self_like(self,defining_string,idx_in_tree=None):
        if idx_in_tree is None:
            idx_in_tree = self.idx_in_tree
        new = LogicalForm(defining_string,idx_in_tree=idx_in_tree,base_lexicon=self.base_lexicon,caches=self.caches,parent=self.parent)
        new.sem_cat = self.sem_cat
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
    def num_lambda_binders(self):
        maybe_lambda_list = split_respecting_brackets(self.subtree_string(),sep='.')
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
            assert is_bracketed(x) or ' ' not in x or self.node_type in ['noun','lmbda','Q']
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
            return 'Q' + bracced_if_var
        elif self.node_type in ['composite','dconst']:
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
        #elif self.node_type == 'const' and as_shell:
        #    return 'PLACE'
        #elif self.node_type == 'noun' and as_shell:
        #    return 'NPLACE'
        #elif self.node_type == 'quant' and as_shell:
        #    return 'QUANT'
        #else:
        #    if as_shell:
        #        print(self.node_type)
        #    x = self.string

        assert as_shell or 'PLACE' not in x
        assert not x.startswith('.')

        return x

    def copy(self):
        copied_version = self.spawn_self_like(self.subtree_string())
        assert copied_version.lf_str_a == self.lf_str_a
        copied_version.sem_cat = self.sem_cat
        return copied_version

    def __repr__(self):
        return (f'LogicalForm of type {self.node_type.upper()}: {self.subtree_string()}'
                f'\n\tsemantic category: {self.sem_cat}\tindex in tree: {self.idx_in_tree}\n')

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
        if not f'${var_num}' in self.subtree_string():
            breakpoint()
        new.stored_subtree_string = f'lambda ${var_num}.{self.subtree_string()}'
        new.stored_alpha_normalized_subtree_string =alpha_normalize(f'lambda ${var_num}.{self.subtree_string()}')
        new.stored_shell_subtree_string = f'lambda ${var_num}.{self.subtree_string(as_shell=True)}'
        new.stored_alpha_normalized_shell_subtree_string = alpha_normalize(f'lambda ${var_num}.{self.subtree_string(as_shell=True)}')
        new.var_descendents = list(set(self.var_descendents + [var_num]))
        return new

    @property
    def sem_cat_is_set(self):
        return self.sem_cat != 'X'

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
        else:
            self.syn_cat = lf.sem_cat
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

    def add_cmp_splits(self,f,g,left_words,right_words):
        # just fwd_cmp for now
        right_child_fwd = ParseNode(g,right_words,parent=self,node_type='right_fwd_cmp')
        new_f_syn_cat, new_g_syn_cat = f_cmp_from_parent_and_g(self.syn_cat,right_child_fwd.syn_cat,sem_only=False)
        assert new_f_syn_cat != 'S/NP\\NP'
        if new_f_syn_cat is None:
            return
        left_child_fwd = ParseNode(f,left_words,parent=self,node_type='left_fwd_cmp',syn_cat=new_f_syn_cat)
        assert is_congruent(f.sem_cat,new_f_syn_cat)
        assert is_congruent(g.sem_cat,new_g_syn_cat)
        right_child_fwd.syn_cat = new_g_syn_cat
        self.append_split(left_child_fwd,right_child_fwd,'fwd_cmp')

    def add_app_splits(self,f,g,left_words,right_words):
            # CONVENTION: f-child with the words on the left of sentence is
            # the 'left' child, even if bck application, left child is g in
            # that case
            right_child_fwd = ParseNode(g,right_words,parent=self,node_type='right_fwd_app')
            new_syn_cat = f'{self.syn_cat}/{maybe_brac(right_child_fwd.syn_cat)}'
            assert new_syn_cat != 'S/NP\\NP'
            left_child_fwd = ParseNode(f,left_words,parent=self,node_type='left_fwd_app',syn_cat=new_syn_cat)
            assert is_congruent(f.sem_cat,new_syn_cat)
            self.append_split(left_child_fwd,right_child_fwd,'fwd_app')

            left_child_bck = ParseNode(g,left_words,parent=self,node_type='left_bck_app')
            new_syn_cat = f'{self.syn_cat}\\{maybe_brac(left_child_bck.syn_cat)}'
            if new_syn_cat != 'S/NP\\NP':
                right_child_bck = ParseNode(f,right_words,parent=self,node_type='right_bck_app',syn_cat=new_syn_cat)
                assert is_congruent(f.sem_cat,new_syn_cat)
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
        sem_cat = maybe_de_type_raise(self.sem_cat)
        syn_cat = maybe_de_type_raise(self.syn_cat)
        if sem_cat != self.sem_cat: # has been de-type-raised
            assert sem_cat in self.sem_cat
            assert syn_cat in self.syn_cat
            assert syn_cat != self.syn_cat
            shell_lf = alpha_normalize(logical_de_type_raise(shell_lf))
            lf = alpha_normalize(logical_de_type_raise(lf))
        return word_str, lf, shell_lf, sem_cat, syn_cat

    def __repr__(self):
        base = (f"ParseNode\n"
                f"\tWords: {' '.join(self.words)}\n"
                f"\tLogical Form: {self.logical_form.subtree_string()}\n"
                f"\tSyntactic Category: {self.syn_cat}\n")
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
            syntax_split = ps['left'].syn_cat + ' + ' + ps['right'].syn_cat
            split_prob = syntax_learner.prob(syntax_split,self.syn_cat)
            left_below_prob = ps['left'].propagate_below_probs(syntax_learner,shell_meaning_learner,meaning_learner,word_learner,cache,split_prob,is_map)
            right_below_prob = ps['right'].propagate_below_probs(syntax_learner,shell_meaning_learner,meaning_learner,word_learner,cache,split_prob,is_map)
            all_probs.append(left_below_prob*right_below_prob*split_prob)

        below_prob = max(all_probs) if is_map else sum(all_probs)
        cache[self] = below_prob
        self.below_prob = below_prob
        return below_prob

    def propagate_above_probs(self,passed_above_prob): #reuse split_prob from propagate_below_probs
        self.above_prob = passed_above_prob*self.split_prob
        for ps in self.possible_splits:
            # this is how to get cousins probs
            ps['left'].propagate_above_probs(self.above_prob*ps['right'].below_prob)
            ps['right'].propagate_above_probs(self.above_prob*ps['left'].below_prob)
            assert abs(ps['left'].above_prob*ps['left'].below_prob - ps['right'].above_prob*ps['right'].below_prob) < 1e-10*(ps['left'].above_prob*ps['left'].below_prob + ps['right'].above_prob*ps['right'].below_prob)
        if self.possible_splits != []:
            if not self.above_prob > max([z.above_prob for ps in self.possible_splits for z in (ps['right'],ps['left'])]):
                breakpoint()

    def prob_as_leaf(self,syntax_learner,shell_meaning_learner,meaning_learner,word_learner):
        word_str, lf, shell_lf, sem_cat, syn_cat = self.info_if_leaf()
        self.stored_prob_as_leaf = syntax_learner.prob('leaf',syn_cat) * \
                                   shell_meaning_learner.prob(shell_lf,sem_cat) * \
                                   meaning_learner.prob(lf,shell_lf) * \
                                   word_learner.prob(word_str,lf)
                            # will use stored value in train_one_step()
        return self.stored_prob_as_leaf
