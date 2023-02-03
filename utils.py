import re


def possible_syn_cats(sem_cat):
    if is_atomic(sem_cat):
        return [sem_cat]
    out_cat,slash,in_cat = cat_components(remove_possible_outer_brackets(sem_cat),'|')
    return [rest_out+sd+rest_in for sd in ('\\','/') for rest_in in possible_syn_cats(in_cat) for rest_out in possible_syn_cats(out_cat)]

def combination_from_sem_cats_and_rule(lsem_cat,rsem_cat,rule):
    if rule == 'fwd_app':
        fin,fslash,fout = cat_components(lsem_cat,'|')
        return f'{fin}/{fout} + {rsem_cat}'
    elif rule == 'bck_app':
        fin,fslash,fout = cat_components(rsem_cat,'|')
        return f'{lsem_cat} + {fin}\\{fout}'
    else:
        breakpoint()

def lambda_body_split(lf):
    """lambda_binder has the dot on the end"""
    try:
        lambda_binder = re.match(r'lambda \$\d+\.',lf).group(0)
    except AttributeError: #there's no lambda binder in the lf
        return '', lf, -1
    body = lf[len(lambda_binder):]
    first_var_num = int(lambda_binder[8:-1])
    return lambda_binder, body, first_var_num

def beta_normalize(m):
    m = remove_possible_outer_brackets(m)
    if m.startswith('lambda'):
        lambda_binder, body, _ = lambda_body_split(m)
        return lambda_binder + beta_normalize(body)
    if re.match(r'^[\w\$]*$',m): # has no variables
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

def strip_string(ss):
    ss = re.sub(r'lambda \$\d+\.','',ss)
    ss = re.sub(r'( ?\$\d+)+$','',ss.replace('(','').replace(')',''))
    # allowed to have trailing bound variables
    return ss.strip()

def concat_lfs(lf1,lf2):
    return '(' + lf1.subtree_string() + ') (' + lf2.subtree_string() + ')'

def is_congruent(sc1,sc2):
    return sc1 == 'XXX' or sc2 == 'XXX' or num_nps(sc1) == num_nps(sc2)

def is_direct_congruent(sc1,sc2):
    sc1 = re.sub(r'[\\/]','|',sc1)
    sc2 = re.sub(r'[\\/]','|',sc2)
    return sc1 == sc2

def num_nps(sem_cat):
    sem_cat = re.sub(r'[\\/]','|',sem_cat)
    if sem_cat == 'NP':
        return 1
    elif is_atomic(sem_cat):
        return 0
    else:
        splits = split_respecting_brackets(sem_cat,sep='|',debracket=True)
        assert splits[0] != sem_cat
        return num_nps(splits[0]) - sum([num_nps(sc) for sc in splits[1:]])

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

def split_respecting_brackets(s,sep=' ',debracket=False):
    """Only make a split when there are no open brackets."""
    if debracket:
        s = remove_possible_outer_brackets(s)
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

def maybe_brac(s):
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
    elif is_atomic(left_cat) or is_atomic(right_cat): # can't do composition then
        return None, None
    elif '|' in left_cat and '|' in right_cat:
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
    else:
        return None,None

def f_cmp_from_parent_and_g(parent_cat,g_cat,sem_only):
    pout,pslash,pin = cat_components(parent_cat,sep=['/','\\','|'])
    gout,gslash,gin = cat_components(g_cat,sep=['/','\\','|'])
    assert remove_possible_outer_brackets(pin) == remove_possible_outer_brackets(gin)
    if sem_only or (gslash != '\\' and pslash != '\\'): # just fwd cmp for now
        return maybe_brac(pout) + gslash + maybe_brac(gout) # hard-coding fwd slash

def parent_cmp_from_f_and_g(f_cat,g_cat,sem_only):
    fout,fslash,fin = cat_components(f_cat,sep=['/','\\','|'])
    gout,gslash,gin = cat_components(g_cat,sep=['/','\\','|'])
    if sem_only or (fslash == '/' and gslash == '/'):
        assert remove_possible_outer_brackets(fin) == remove_possible_outer_brackets(gout)
        return maybe_brac(fout) + fslash + maybe_brac(gin)

def alpha_normalize(x):
    trans_list = sorted(set(re.findall(r'\$\d{1,2}',x)))
    for v_new, v_old in enumerate(trans_list):
        x = x.replace(fr'{v_old}',f'${v_new}')
    return x

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

