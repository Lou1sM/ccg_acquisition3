import re


def is_balanced_nums_brackets(s):
    return len(re.findall(r'\(',s)) == len(re.findall(r'\)',s))

def outermost_first_bracketed_chunk(s):
    """Similar to the first argument returned by split_respecting_brackets,
    but here we don't need a separator, just split as soon as brackets are
    closed, e.g. f(x,y)g(z) --> f(x,y)
    """
    if '(' not in s:
        assert ')' not in s
        return s
    num_open_brackets = 0
    has_been_bracketed = False
    for i,c in enumerate(s):
        if c == '(':
            num_open_brackets += 1
            has_been_bracketed = True
        elif c == ')':
            num_open_brackets -= 1
        if num_open_brackets == 0 and has_been_bracketed:
            assert is_balanced_nums_brackets(s[:i+1])
            assert is_balanced_nums_brackets(s[i+1:])
            return s[:i+1], s[i+1:]
    breakpoint()

def split_respecting_brackets(s,sep=' '):
    """Only make a split when there are no open brackets."""
    num_open_brackets = 0
    split_points = [-1]
    if isinstance(sep,str):
        sep = [sep]
    else:
        assert isinstance(sep,list)

    for i,c in enumerate(s):
        if c in sep and num_open_brackets == 0:
            split_points.append(i)
        elif c == '(':
            num_open_brackets += 1
        elif c == ')':
            num_open_brackets -= 1
    split_points.append(len(s))
    splits = [s[split_points[i]+1:split_points[i+1]] for i in range(len(split_points)-1)]
    return splits

def is_bracketed(s):
    return s.startswith('(') and s.endswith(')')

def maybe_bracketted(s):
    """Add brackets if not a leaf."""
    if '|' in s or '\\' in s or '/' in s:
        return '('+s+')'
    else:
        return s

def all_sublists(x):
    if len(x) == 0:
        return [[]]

    recursed = all_sublists(x[1:])
    return recursed + [[x[0]]+item for item in recursed]

def remove_possible_outer_brackets(s):
    while True:
        if not is_bracketed(s):
            return s
        elif outermost_first_bracketed_chunk(s)[0] == s:
            s = s[1:-1]
        else:
            return s

def normalize_dict(d):
    if isinstance(d,list):
        assert all([isinstance(item,tuple) for item in d])
        norm = sum([item[1] for item in d])
        return [(item[0],item[1]/norm) for item in d]
    else:
        assert isinstance(d,dict)
        norm = sum(d.values())
        return {k:v/norm for k,v in d.items()}

def translate_by_unify(x,y):
    """Equate corresponding subexpressions in x and y to form a translation between subexpressions"""
    translation = {}
    x_list = re.split(r'[ ,().]',x)
    y_list = re.split(r'[ ,().]',y)
    assert len(x_list) == len(y_list)
    new_var_num_diff = 'none'
    for x_term,y_term in zip(x_list,y_list):
        if 'lambda' not in x_term:
            assert 'lambda' not in y_term
            if '$' in x_term:
                assert '$' in y_term
                new_new_var_num_diff = int(y_term[1:]) - int(x_term[1:])
                assert new_var_num_diff in ['none',new_new_var_num_diff]
                new_var_num_diff = new_new_var_num_diff
            if x_term!=y_term:
                translation[x_term] = y_term
    return translation

def translate(s,trans_dict):
    if len(trans_dict)==0:
        return s
    var_trans_sources = [x for x in trans_dict.keys() if '$' in x]
    var_trans_targets = [x for x in trans_dict.values() if '$' in x]
    if var_trans_sources != []:
        max_var_in_source_language = max(var_trans_sources, key=lambda x: int(x[1:]))
        max_var_in_target_language = max(var_trans_targets, key=lambda x: int(x[1:]))
        new_var_num_diff = int(max_var_in_target_language[1:])-int(max_var_in_source_language[1:])
        for var in re.findall(r'\$\d',s):
            if var not in trans_dict:
                trans_dict[var] = f'${int(var[1:])+new_var_num_diff}'
    trans_dict1 = {k:v[0]+'£'+''.join(v[1:]) for k,v in trans_dict.items()}
    trans_dict2 = {k[0]+'£'+''.join(k[1:]):k for k in trans_dict.values()}
    for old,new in trans_dict1.items():
        s = s.replace(old,new)
    for old,new in trans_dict2.items():
        s = s.replace(old,new)
    return s

def simple_translate(s,trans_dict):
    for old,new in trans_dict.items():
        s = s.replace(old,new)
    return s

def file_print(s,f):
    print(s)
    print(s,file=f)

def is_atomic(cat):
    return '\\' not in cat and '/' not in cat and '|' not in cat

def get_combination(left_cat,right_cat):
    """Inputs can be either syncats or semcats"""
    if is_atomic(left_cat) and is_atomic(right_cat):
        return None, None
    elif left_cat.endswith('('+right_cat+')'):
        return left_cat[:-len(right_cat)-3],'fwd_app'
    elif left_cat.endswith(right_cat) and is_atomic(right_cat):
        return left_cat[:-len(right_cat)-1],'fwd_app'
    elif right_cat.endswith('('+left_cat+')'):
        return right_cat[:-len(left_cat)-3],'bck_app'
    elif right_cat.endswith(left_cat) and is_atomic(left_cat):
        return right_cat[:-len(left_cat)-1],'bck_app'
    else:
        left_out, left_slash, left_in = cat_components(left_cat,sep='|')
        right_out, right_slash, right_in = cat_components(right_cat,sep='|')
        if left_slash != right_slash and '|' not in [left_slash ,right_slash]: # skip crossed composition
            return None, None
        elif left_in == right_out:
            return ''.join(left_out, left_slash, right_in), 'fwd_comp'
        elif left_out == right_in:
            return ''.join(right_out, right_slash, left_in), 'bck_comp'
        else:
            return None,None

def maybe_app(sc1,sc2,direction):
    if direction=='bck':
        slash = '\\'
        applier = sc2
        appliee = sc1
    else:
        slash = '/'
        applier = sc1
        appliee = sc2
    if appliee in applier:
        if is_atomic(appliee):
            if applier.endswith(f'{slash}{appliee}'):
                return applier[:-len(appliee)-1]
        else:
            if applier.endswith(f'{slash}({appliee})'):
                return applier[:-len(appliee)-3]

def cat_components(syn_cat,sep):
    splits = split_respecting_brackets(syn_cat,sep=sep)
    in_cat = splits[-1]
    slash = syn_cat[-len(in_cat)-1]
    out_cat = syn_cat[:-len(in_cat)-1]
    return out_cat, slash, in_cat

def get_combination_old(left_syn_cat,right_syn_cat):
    if is_atomic(left_syn_cat) and is_atomic(right_syn_cat):
        return None
    if left_syn_cat in right_syn_cat: # can only be bck app then
        return maybe_app(left_syn_cat,right_syn_cat,direction='bck')
    elif right_syn_cat in left_syn_cat: # can only be fwd app then
        return maybe_app(left_syn_cat,right_syn_cat,direction='fwd')
    else: # see if works by composition
        left_out, left_slash, left_in = cat_components(left_syn_cat,sep=['\\','/'])
        right_out, right_slash, right_in = cat_components(right_syn_cat,sep=['\\','/'])
        if left_slash != right_slash: # skip crossed composition
            return None
        elif left_in == right_out:
            return ''.join(left_out, left_slash, right_in)
        elif left_out == right_in:
            return ''.join(right_out, right_slash, left_in)

def parses_of_syn_cats(self,syn_cats):
    """Return all possible parses (often only one) of the given syn_cats in the given order."""
    frontiers = [[syn_cats]]
    for _ in range(len(syn_cats)-1):
        frontiers = [current+[f] for current in frontiers for f in possible_next_frontiers(current[-1])]
    return frontiers

