# this should make the graph files
# for each of the syntactic phenomona
# needed.

# S,V,O
# P,NP
# Det,N
# Adj,Num
# Num,N

# Wh-init vs in situ
# Adv,Vfin
# prodrop

# input will be, sem_type and list of cats that
# this sem_type can take


# 1. get cat prob from grammar.
def cat_prob_from_grammar(sc,rule_set):
    c_prob = 1.0
    rule_prob = 1.0
    if sc.atomic(): return 1.0
    elif sc.direction=="fwd":
        rule = (sc.funct.to_string(),sc.to_string()+'#####'+sc.arg.to_string())
        if rule_set.check_target(rule[1]):
            rule_prob = rule_set.return_prob(rule[0],rule[1])
        c_prob = c_prob*rule_prob
        c_prob = c_prob*cat_prob_from_grammar(sc.funct,rule_set)
    elif sc.direction=="back":
        rule = (sc.funct.to_string(),sc.arg.to_string()+'#####'+sc.to_string())
        if rule_set.check_target(rule[1]):
            rule_prob = rule_set.return_prob(rule[0],rule[1])
        c_prob = c_prob*rule_prob
        c_prob = c_prob*cat_prob_from_grammar(sc.funct,rule_set)

    return c_prob

# 2. get P(lf_{type}|cat) from lexicon.
def get_sem_type_prob(lexicon,sem_store,cat,lf_type,pos_type,arity,varorder):
    catkey = cat.to_string()
    if catkey in lexicon.syntax:
        sem_typeprobs_ = lexicon.get_sem_from_type_prob(sem_store,catkey,lf_type,pos_type,arity,varorder)
        vo = ""
        for v in varorder:
            vo = vo+str(v)
            dictkey = (lf_type,pos_type,arity,vo)

        if dictkey in sem_typeprobs_:
            return sem_typeprobs_[dictkey]
    return 0.0

# 3. get P(cat|lf_{type},postype).
def get_pcat_given_lf_type(cat,lf_type,pos_type,arity,varorder,lexicon,sem_store,rule_set,out_file):
    p_c = cat_prob_from_grammar(cat,rule_set)
    p_l_given_c = get_sem_type_prob(lexicon,sem_store,cat,lf_type,pos_type,arity,varorder)
    print(p_c,":",p_l_given_c,"  ",file=out_file)
    p_c_given_l = p_l_given_c*p_c # /pL
    return p_c_given_l

# 4. get all probs to be compared
def output_cat_probs(pos_type,lf_type,arity,cats,lexicon,sem_store,rule_set,out_file):
    norm = 0.0
    probList = []
    for (cat,varorder) in cats:
        arity = len(varorder)
        p_c_given_l = get_pcat_given_lf_type(cat,lf_type,pos_type,arity,varorder,lexicon,sem_store,rule_set,out_file)
        norm += p_c_given_l
        probList.append(p_c_given_l)
    print('', file=out_file)
    for p in probList:
        if norm==0.0: print(1.0/len(probList),' ',file=out_file)
        else: print(out_file,p/norm,' ',file=out_file)
    print('',file=out_file)
