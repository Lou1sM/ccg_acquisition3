import re


LAMBDA_RE_STR = r'^lambda \$\d{1,2}(_\{(e|r|<r,t>|<<e,e>,e>)\})?\.'
#LAMBDA_RE_STR = r'^lambda (WHAT1|WHO1|\$\d{1,2}|\$\d{1,2}_\{e\}|\$\d{1,2}_\{r\}|\$\d{1,2}_\{<r,t>\}|\$\d{1,2}\{<<e,e>,e>\})?\.'
def de_q(lf):
    lambda_binder, body = all_lambda_body_splits(lf)
    if body.startswith('Q '):
        body = maybe_debrac(body[2:])
    return lambda_binder + body

def lf_acc(lf_pred, lf_gt):
    return de_q(lf_pred) == de_q(lf_gt)

def new_var_num(lf_str):
    vars_in_self = re.findall(r'\$\d{1,2}',lf_str)
    if len(vars_in_self) == 0:
        return 0
    return max([int(x[1:]) for x in vars_in_self])+1

def lambda_match(maybe_lambda_str):
    if not maybe_lambda_str.startswith('lambda '):
        return None # fast check without re to rule out most
    #return re.match(r'^lambda \$\d{1,2}(_\{(e|r|<r,t>|<<e,e>,e>)\})?\.',maybe_lambda_str)
    return re.match(LAMBDA_RE_STR, maybe_lambda_str)

def is_wellformed_lf(lf, should_be_normed=False):
    if lf == '': return True
    if bool(re.search(r'lambda(?! \$\d)',lf)): # rule out this simple string
    #if bool(re.search(r'lambda(?! (WHAT1|WHO1|\$\d))', lf)):
        return False
    if lf.count('.') != lf.count('lambda'):
        return False
    if should_be_normed:
        if not lf.startswith('Q') and bool(re.search(r'(?<!\.)Q', lf)): # Q should only be at beginning
            return False
        lambdas, body = all_lambda_body_splits(lf)
        if lf.count('.') != lambdas.count('.'):
            return False
    lf = maybe_debrac(lf)
    if bool(re.match(r'[a-zA-Z0-9_]+',lf)):
        return True
    if bool(re.match(r'\$\d{1,2}$',lf)):
        return True
    if bool(possible_first_lambda := lambda_match(lf)):
        return is_wellformed_lf(lf[possible_first_lambda.end():])
    splits = split_respecting_brackets(lf)
    if len(splits) > 1:
        return all([is_wellformed_lf(s) for s in splits])
    return False

def maybe_de_type_raise(cat):
    splits = split_respecting_brackets(cat, sep='|')
    incat = splits[-1]
    outcat = '|'.join(splits[:-1])
    insplits = split_respecting_brackets(incat, sep='|')
    if len(insplits)==2 and insplits[0]==outcat:
        return insplits[1]
    else:
        return cat
    #new_cat = re.sub(r'([A-Z]{1,2})[\\/\|]\(\1[\\/\|](.*)\)$',r'\2',cat)
    #return new_cat

def combine_lfs(f_str,g_str,comb_type,normalize=True):
    increment_g_vars_by = new_var_num(f_str)
    vars_in_g = re.findall(r'(?<=\$)\d{1,2}',g_str)
    orig_g_str = alpha_normalize(g_str)
    if increment_g_vars_by > 0:
        for v in sorted(set(vars_in_g),reverse=True):
            #g_str = g_str.replace(f'${v}',f'${int(v)+increment_g_vars_by}')
            g_str = re.sub(rf'\${v}(?!\d)',f'${int(v)+increment_g_vars_by}',g_str)
    assert alpha_normalize(g_str) == orig_g_str
    if comb_type == 'app':
        unnormed = concat_lfs(f_str,g_str)
    elif comb_type == 'cmp':
        unnormed = get_cmp_of_lfs(f_str,g_str)
    else:
        breakpoint()
    assert '$$' not in unnormed
    return alpha_normalize(beta_normalize(unnormed)) if normalize else unnormed

def logical_type_raise(lf_str):
    n = new_var_num(lf_str)
    return f"lambda ${n}.${n} {maybe_brac(lf_str,sep=' ')}"

def is_cat_type_raised(sem_cat):
    splits = split_respecting_brackets(sem_cat, sep='|')
    if len(splits) != 2:
        return False
    out_cat, in_cat = splits
    in_cat_splits = split_respecting_brackets(in_cat, sep='|')
    if len(in_cat_splits) != 2:
        return False
    return in_cat_splits[0] == out_cat

def is_type_raised(lf_str):
    if not bool(possible_first_lambda := lambda_match(lf_str)):
        return False
    #first_lambda_var_num = lf_str[possible_first_lambda.end()-2:possible_first_lambda.end()]
    first_lambda_var_num = possible_first_lambda.group()[7:-1]
    body = split_respecting_brackets(lf_str,sep='.')[-1]
    return body.startswith(first_lambda_var_num)

def logical_de_type_raise(lf_str):
    lambda_binder, _, rest = lf_str.rpartition('.')
    #type_raising_part = re.match(r'lambda (\$\d{1,2}).\1',lf_str)
    type_raising_part = lambda_match(lf_str)
    len_of_type_raising_part = type_raising_part.span()[1]
    rest = lf_str[len_of_type_raising_part:]
    maybe_lambda, body = all_lambda_body_splits(rest)
    if not body.startswith('$0'):
        return lf_str
    return maybe_lambda + body[3:]
    rest = maybe_debrac(rest)
    if not is_wellformed_lf(rest):
        breakpoint()
    return rest

def get_cmp_of_lfs(f_str,g_str):
    var_nums = re.findall(r'\$\d{1,2}',f_str+g_str)
    new_var_num = max([int(x[1:]) for x in var_nums])+1
    return f'lambda ${new_var_num}.({f_str}) (({g_str}) ${new_var_num})'

def old_get_cmp_of_lfs(f_str,g_str):
    f_lambda_binder,frest,f_first_var_num = first_lambda_body_split(f_str)
    _,grest,g_first_var_num = first_lambda_body_split(g_str)
    #max_var_num = max(f_str.new_var_num,g_str.new_var_num)
    var_nums = re.findall(r'\$\d{1,2}',f_str+g_str)
    new_var_num = max([int(x[1:]) for x in var_nums])+1
    f_lambda_binder = f_lambda_binder.replace(f'${f_first_var_num}',f'${new_var_num}')
    to_sub_in = grest.replace(f'${g_first_var_num}',f'${new_var_num}')
    subbed_in = frest.replace(f'${f_first_var_num}',f'({to_sub_in})')
    if not is_wellformed_lf(f_lambda_binder+subbed_in):
        breakpoint()
    return f_lambda_binder + subbed_in

def possible_syncs(sem_cat):
    if is_atomic(sem_cat):
        return [sem_cat]
    out_cat,slash,in_cat = cat_components(maybe_debrac(sem_cat),'|')
    return [rest_out+sd+maybe_brac(rest_in) for sd in ('\\','/') for rest_in in possible_syncs(in_cat) for rest_out in possible_syncs(out_cat)]

def apply_sem_cats(fsc, gsc):
    hits = re.findall(fr'[\\/\|]\(?{re.escape(gsc)}\)?$',fsc.replace('\\','\\\\'))
    assert len(hits) < 2
    if len(hits)==0:
        return None
    hit = hits[0][1:]
    if not is_bracket_balanced(hit):
        return None
    if '|' in hit and not is_bracketed(hit):
        return None
    psc = fsc[:-len(hits[0])]
    return psc

def combination_from_sem_cats_and_rule(lsem_cat,rsem_cat,rule):
    if rule == 'fwd_app':
        fin,fslash,fout = cat_components(lsem_cat,'|')
        return f'{fin}/{fout} + {rsem_cat}'
    elif rule == 'bck_app':
        fin,fslash,fout = cat_components(rsem_cat,'|')
        return f'{lsem_cat} + {fin}\\{fout}'
    else:
        breakpoint()

def first_lambda_body_split(lf):
    """lambda_binder has the dot on the end"""
    try:
        #lambda_binder = re.match(r'^lambda \$\d{1,2}(_\{(e|r|<r,t>)\})?\.',lf).group(0)
        lambda_binder = lambda_match(lf).group(0)
    except AttributeError: #there's no lambda binder in the lf
        return '', lf, -1
    body = lf[len(lambda_binder):]
    first_var_num = int(lambda_binder[8:lambda_binder.index('_')] if '_' in lambda_binder else lambda_binder[8:-1])
    return lambda_binder, body, first_var_num

def all_lambda_body_splits(lf):
    """lambda_binder has the dot on the end"""
    try:
        #lambda_binder = re.match(r'(lambda \$\d{1,2}?+\.)*',lf).group(0)
        lambda_binder = re.match(r'(lambda \$\d{1,2}(_\{[er]\})?\.)*',lf).group(0)
    except AttributeError: #there's no lambda binder in the lf
        return '', lf
    body = lf[len(lambda_binder):]
    return lambda_binder, body

def alpha_normalize(x):
    hits = re.findall(r'(?<=\$)\d{1,2}',x)
    if len(hits) == 0:
        return x
    trans_list = sorted(set(hits),key=lambda x:hits.index(x))
    #buffer = int(max(trans_list))+1 # big enough to not clobber anything
    buffer = max([int(x) for x in trans_list]) + 1
    #if x == 'lambda $2.Q (mod|do_2 (v|think_4 pro:per|you_3 (lambda $5_{r}.v|say-3s_6 pro:per|it_5 $2 $1)))':
        #breakpoint()
    for v_new, v_old in enumerate(trans_list):
        assert f'${v_new+buffer}' not in x
        #x = x.replace(fr'${v_old}',f'${v_new+buffer}')
        x = re.sub(rf'\${v_old}(?!\d)',f'${v_new+buffer}',x)
    for v_num in range(len(trans_list)):
        assert not bool(re.search(fr'${v_num}',x))
        #x = x.replace(fr'${v_num+buffer}',f'${v_num}')
        x = re.sub(rf'\${v_num+buffer}(?!\d)',f'${v_num}',x)
    return x

def beta_normalize(m,verbose=False):
    m = maybe_debrac(m)
    assert is_wellformed_lf(m)
    if m.startswith('lambda'):
        lambda_binder, body, _ = first_lambda_body_split(m)
        assert lambda_binder != ''
        return lambda_binder + beta_normalize(body)
    if re.match(r'^[\w\$]*$',m): # has no variables
        if verbose: print(m)
        return m
    splits = split_respecting_brackets(m)
    if len(splits) == 1:
        if verbose: print(m)
        return m
    left_ = maybe_debrac(' '.join(splits[:-1]))
    left = beta_normalize(left_)
    assert 'lambda (' not in left
    right_ = splits[-1]
    right = beta_normalize(right_)
    #if not re.match(r'^[\w\$\-]*$',right) and not is_bracketed(right):
        #right = '('+right+')'
    right = maybe_brac(right, sep=' ')

    if left.startswith('lambda'):
        lambda_binder,_,rest = left.partition('.')
        #assert re.match(r'lambda \$\d{1,2}',lambda_binder)
        assert lambda_match(lambda_binder+'.')
        var_name = lambda_binder[7:]
        assert re.match(r'\$\d',var_name)
        combined = re.sub(re.escape(f'({var_name})'),right,rest) # to avoid doubly-bracketted
        combined = re.sub(re.escape(var_name),right,combined)
        normed = beta_normalize(combined)
    else:
        normed = ' '.join([left,right])
    if verbose: print(normed)
    normed = normed.strip()
    return normed

def strip_string(ss):
    #ss = re.sub(r'(lambda \$\d{1,2})+\.','', ss)
    ss = re.sub(r'^(lambda \$\d{1,2}\.)+','', ss)
    #ss = re.sub(r'( ?\$\d+)+$','',ss.replace('(','').replace(')',''))
    ss = re.sub(r' ?\$\d{1,2}','', ss)
    ss = re.sub(r'\( *\)','', ss)
    # allowed to have trailing bound variables
    return ss.strip()

def concat_lfs(lf_str1,lf_str2):
    return '(' + lf_str1 + ') (' + lf_str2 + ')'

def set_congruent(sc1s, sc2s):
    return set(x for x in sc1s if any(is_congruent(x,y) for y in sc2s))

def is_congruent(sc1,sc2):
    if sc1 is None or sc2 is None:
        return False
    return 'X' in sc1 or 'X' in sc2 or n_nps(sc1) == n_nps(sc2)

def is_direct_congruent(sc1,sc2):
    sc1 = re.sub(r'[\\/]','|',sc1)
    sc2 = re.sub(r'[\\/]','|',sc2)
    return sc1 == sc2

def n_nps(sem_cat):
    if sem_cat == 'NP':
        return 1
    elif sem_cat == 'VP':
        return -1
    elif sem_cat in ['Sq','S','N','Swhq']:
        return 0
    else:
        splits = split_respecting_brackets(sem_cat,sep=['\\','/','|'],debracket=True)
        if not splits[0] != sem_cat:
            breakpoint()
        return n_nps(splits[0]) - sum([n_nps(sc) for sc in splits[1:]])

def is_bracket_balanced(s):
    if len(re.findall(r'\(',s)) != len(re.findall(r'\)',s)):
        return False
    return split_respecting_brackets(s) != []

def outermost_first_chunk(s):
    """Similar to the first argument returned by split_respecting_brackets,
    but here we don't need a separator, just split as soon as brackets are
    closed, e.g. f(x,y)g(z) --> f(x,y)
    """
    if '(' not in s:
        assert ')' not in s
        return s, ''
    #if not s.startswith('('):
        #return s[:s.index('(')],s[s.index('('):]
    n_open_brackets = 0
    has_been_bracketed = False
    for i,c in enumerate(s):
        if c == '(':
            n_open_brackets += 1
            has_been_bracketed = True
        elif c == ')':
            n_open_brackets -= 1
        if n_open_brackets == 0 and has_been_bracketed:
            assert is_bracket_balanced(s[:i+1])
            assert is_bracket_balanced(s[i+1:])
            return s[:i+1], s[i+1:]
    breakpoint()

def split_respecting_brackets(s,sep=' ',debracket=False):
    """Only make a split when there are no open brackets."""
    if debracket:
        s = maybe_debrac(s)
    n_open_brackets = 0
    split_points = [-1]
    if isinstance(sep,str):
        sep = [sep]
    else:
        assert isinstance(sep,list)

    for i,c in enumerate(s):
        if c in sep and n_open_brackets == 0:
            split_points.append(i)
        elif c == '(':
            n_open_brackets += 1
        elif c == ')':
            n_open_brackets -= 1
        if n_open_brackets < 0:
            return []
    split_points.append(len(s))
    splits = [s[split_points[i]+1:split_points[i+1]] for i in range(len(split_points)-1)]
    return splits

def is_bracketed(s):
    return s.startswith('(') and s.endswith(')')

def maybe_brac(s,sep=['|','\\','/']):
    """Add brackets if not a leaf."""
    #if '|' in s or '\\' in s or '/' in s:
    if not any([x in s for x in sep]):
        return s
    elif len(split_respecting_brackets(s,sep=sep)) > 1:
        return '('+s+')'
    else:
        return s

def maybe_debrac(s):
    while True:
        if not is_bracketed(s):
            return s
        elif outermost_first_chunk(s)[0] == s: # don't debrac cases like (...)(...)
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

def n_lambda_binders(s):
    if '.' not in s:
        return 0
    maybe_lambda_list = split_respecting_brackets(s,sep='.')
    lambdas, body = maybe_lambda_list[:-1], maybe_lambda_list[-1]
    assert all(x.startswith('lambda') for x in lambdas)
    #maybe_lambda_list = set(x for x in maybe_lambda_list_ if not x.endswith('_{e}'))
    #maybe_lambda_list = [m for m in maybe_lambda_list if m.startswith('lambda')]
    #if bool(leading_vars := re.match(r'(\$\d{1,2} ?)*', body)):# don't count type-raised vars
        #type_raised_vars = leading_vars.group().split()
        #lambdas = [m for m in lambdas if all(trv not in m for trv in type_raised_vars)]
    return len(lambdas)

def n_lambda_binders_old(s):
    if '.' not in s:
        return 0
    maybe_lambda_list = split_respecting_brackets(s,sep='.')
    lambdas, body = maybe_lambda_list[:-1], maybe_lambda_list[-1]
    assert all(x.startswith('lambda') for x in lambdas)
    #maybe_lambda_list = set(x for x in maybe_lambda_list_ if not x.endswith('_{e}'))
    #maybe_lambda_list = [m for m in maybe_lambda_list if m.startswith('lambda')]
    if bool(leading_vars := re.match(r'(\$\d{1,2} ?)*', body)):# don't count type-raised vars
        type_raised_vars = leading_vars.group().split()
        lambdas = [m for m in lambdas if all(trv not in m for trv in type_raised_vars)]
    return len(lambdas)

def translate_by_unify(x,y):
    """Equate corresponding subexpressions in x and y to form a translation between subexpressions"""
    translation = {}
    x_list = re.split(r'[ ,().]',x)
    y_list = re.split(r'[ ,().]',y)
    assert len(x_list) == len(y_list)
    new_var_n_diff = 'none'
    for x_term,y_term in zip(x_list,y_list):
        if 'lambda' not in x_term:
            assert 'lambda' not in y_term
            if '$' in x_term:
                assert '$' in y_term
                new_new_var_n_diff = int(y_term[1:]) - int(x_term[1:])
                assert new_var_n_diff in ['none',new_new_var_n_diff]
                new_var_n_diff = new_new_var_n_diff
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
        new_var_n_diff = int(max_var_in_target_language[1:])-int(max_var_in_source_language[1:])
        for var in re.findall(r'\$\d',s):
            if var not in trans_dict:
                trans_dict[var] = f'${int(var[1:])+new_var_n_diff}'
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

def is_fit_by_type_raise(left_cat,right_cat):
    if is_atomic(right_cat): # just fwd for now
        return False, None
    rout,rslash,rin = cat_components(right_cat)
    if is_atomic(rout): # just fwd for now
        return False, None
    routout,routslash,routin = cat_components(rout)
    return routin==left_cat, routout

def non_directional(cat):
    return cat.replace('\\','|').replace('/','|')

def get_combination(left_cat,right_cat):
    """Inputs can be either syncats or semcats"""
    if is_atomic(left_cat) and is_atomic(right_cat):
        return None, None
    elif re.search(fr'[/|]\({re.escape(right_cat)}\)$',non_directional(left_cat)):
        combined, rule = left_cat[:-len(right_cat)-3],'fwd_app'
    elif re.search(fr'[/|]{re.escape(right_cat)}$',non_directional(left_cat)) and is_atomic(right_cat):
        combined, rule = left_cat[:-len(right_cat)-1],'fwd_app'
    elif re.search(fr'[\\|]\({re.escape(left_cat)}\)$',non_directional(right_cat)):
        combined, rule = right_cat[:-len(left_cat)-3],'bck_app'
    elif re.search(fr'[\\|]{re.escape(left_cat)}$',non_directional(right_cat)) and is_atomic(left_cat):
        combined, rule = right_cat[:-len(left_cat)-1],'bck_app'
    elif is_atomic(left_cat) or is_atomic(right_cat): # can't do composition then
        return None, None
    else:
        left_out, left_slash, left_in = cat_components(left_cat)
        right_out, right_slash, right_in = cat_components(right_cat)
        if left_slash != right_slash and '|' not in [left_slash ,right_slash]: # skip crossed composition
            return None, None
        elif maybe_debrac(left_in) == maybe_debrac(right_out):
            combined, rule = ''.join((left_out, left_slash, right_in)), 'fwd_cmp'
        elif maybe_debrac(left_out) == maybe_debrac(right_in):
            combined, rule = ''.join((right_out, right_slash, left_in)), 'bck_cmp'
        #elif non_directional(left_in) == right_in or non_directional(right_in) == left_in:
            #combined, rule = ''.join((right_out, right_slash, left_in)), 'bck_cmp' what was I doing with this?
        else:
            return None,None
    if combined in ['S|N','S|N|NP']:
        return None, None
    else:
        return combined, rule

def infer_slash(lcat,rcat,parent_cat,rule):
    if rule == 'fwd_app':
        to_remove = len(rcat)+1 if is_atomic(rcat) else len(rcat)+3
        out_lcat = lcat[:-to_remove]
        assert is_direct_congruent(out_lcat, parent_cat)
        #if is_atomic(rcat):
            #assert is_direct_congruent(lcat[:-len(rcat)-1],parent_cat)
        #else:
            #assert is_direct_congruent(lcat[:-len(rcat)-3],parent_cat)
        inferredlcat = out_lcat + '/' + maybe_brac(rcat)
        inferredrcat = rcat
    elif rule == 'bck_app':
        to_remove = len(lcat)+1 if is_atomic(lcat) else len(lcat)+3
        out_rcat = rcat[:-to_remove]
        assert is_direct_congruent(out_rcat, parent_cat)

        #if is_atomic(lcat):
            #assert is_direct_congruent(rcat[:-len(lcat)-1],parent_cat)
        #else:
            #assert is_direct_congruent(rcat[:-len(lcat)-3],parent_cat)
        inferredrcat = out_rcat + '\\' + lcat
        inferredlcat = lcat
    else:
        left_out, left_slash, left_in = cat_components(lcat)
        right_out, right_slash, right_in = cat_components(rcat)
        assert left_slash == right_slash
    if rule == 'fwd_cmp':
        inferredlcat = left_out + '/' + right_in
        inferredrcat = rcat
    elif rule == 'bck_cmp':
        inferredrcat = right_out + '\\' + left_in
        inferredlcat = lcat

    elif '\\' not in inferredlcat and '/' not in inferredlcat and '\\' not in inferredrcat and '/' not in inferredrcat:
        breakpoint()
    return inferredlcat, inferredrcat

def f_cmp_from_parent_and_g(parent_cat,g_cat,sem_only):
    """Determines the slash directions for both f and g when comb. with fwd_cmp."""
    if parent_cat=='X' or g_cat=='X':
        return 'X', 'X'
    if is_atomic(g_cat) or is_atomic(parent_cat):
        return None, None
    try:
        pout,pslash,pin = cat_components(parent_cat)
        gout,gslash,gin = cat_components(g_cat)
    except ValueError:
        breakpoint()
    #assert maybe_debrac(pin) == maybe_debrac(gin)
    #if is_atomic(gout): # only consider if g is type-raised and in that case has slash
        #return None, None
    #elif gslash == '\\' or pslash == '\\': # disallow bck cmp for now
        #return None, None
    #elif sem_only:
    if gslash not in ['|', pslash]:
        return None, None
    if pin != gin:
        return None, None
    if sem_only:
        assert pslash == '|' and gslash == '|'
        return maybe_brac(pout) + '|' + maybe_brac(gout), g_cat # hard-coding fwd slash
    else:
        #goutout,gout_slash,goutin = cat_components(gout,allow_atomic=True)
        #composed_cat = goutout + '\\' + goutin # has to be mirror of the main slash
        new_f = pout + pslash + maybe_brac(gout)
        new_g = gout + pslash + gin
        return new_f, new_g

def lf_cat_congruent(lf_str, sem_cat):
    #sem_cat = maybe_de_type_raise(sem_cat_)
    assert 'VP' not in sem_cat
    assert ' you' not in lf_str
    if sem_cat in ('S|N', 'NP|NP'):
        return False
    sem_cat = re.sub(r'^VP','S|NP',sem_cat) # if VP on left then no bracks because right-assoc
    if sem_cat in ('Swhq','N|N', 'N\\N','N/N'):
        what_n_lambdas_should_be = 0
    else:
        what_n_lambdas_should_be = len(split_respecting_brackets(sem_cat,sep=['|','\\','/']))-1
    #return what_n_lambdas_should_be == n_lambda_binders(lf_str) + (' you' in lf_str)
    #if what_n_lambdas_should_be != n_lambda_binders(lf_str) and what_n_lambdas_should_be == n_lambda_binders_old(lf_str):
        #breakpoint()
    return what_n_lambdas_should_be == n_lambda_binders(lf_str)

#def lf_cat_congruent(lf_str, sem_cat):
#    if _lf_cat_congruent(lf_str, sem_cat):
#        return True
#    if is_type_raised(sem_cat):
#        return _lf_cat_congruent(lf_str, maybe_de_type_raise(sem_cat))
#    else:
#        return False

def parent_cmp_from_f_and_g(f_cat, g_cat, sem_only):
    if f_cat=='X' or g_cat=='X':
        return 'X'
    fout,fslash,fin = cat_components(f_cat,sep=['/','\\','|'])
    gout,gslash,gin = cat_components(g_cat,sep=['/','\\','|'])
    if sem_only or (fslash == '/' and gslash == '/'):
        domatch = (maybe_debrac(fin) == maybe_debrac(gout))
        return fout + fslash + maybe_brac(gin) if domatch else None

def reverse_slash(slash):
    assert slash in ['\\','/']
    return '\\' if slash == '/' else '/'

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

def cat_components(sync,allow_atomic=False,sep=None):
    if is_atomic(sync):
        assert allow_atomic
        return sync
    if sep is None:
        sep = ['\\','/','|']
    splits = split_respecting_brackets(sync,sep=sep)
    in_cat = splits[-1]
    slash = sync[-len(in_cat)-1]
    out_cat = sync[:-len(in_cat)-1]
    return out_cat, slash, in_cat

def get_combination_old(left_sync,right_sync):
    if is_atomic(left_sync) and is_atomic(right_sync):
        return None
    if left_sync in right_sync: # can only be bck app then
        return maybe_app(left_sync,right_sync,direction='bck')
    elif right_sync in left_sync: # can only be fwd app then
        return maybe_app(left_sync,right_sync,direction='fwd')
    else: # see if works by composition
        left_out, left_slash, left_in = cat_components(left_sync,sep=['\\','/'])
        right_out, right_slash, right_in = cat_components(right_sync,sep=['\\','/'])
        if left_slash != right_slash: # skip crossed composition
            return None
        elif left_in == right_out:
            return ''.join(left_out, left_slash, right_in)
        elif left_out == right_in:
            return ''.join(right_out, right_slash, left_in)

def balanced_substrings(s):
    open_bracs_idxs = []
    bss = []
    for i,c in enumerate(s):
        if c=='(':
            open_bracs_idxs.append(i)
        elif c==')':
            bss.append(s[open_bracs_idxs.pop():i+1])
    return bss

def nth_in_list(l,n,item):
    return [i for i,x in enumerate(l) if x==item][n]

def all_sublists(x):
    if len(x) == 0:
        return [[]]

    recursed = all_sublists(x[1:])
    return recursed + [[x[0]]+item for item in recursed]

