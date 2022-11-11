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
def catProbFromGrammar(sc,RuleSet):
    c_prob = 1.0
    rule_prob = 1.0
    if sc.atomic(): return 1.0
    elif sc.direction=="fwd":
        # fwd
        rule = (sc.funct.toString(),sc.toString()+'#####'+sc.arg.toString())
        if RuleSet.check_target(rule[1]):
            rule_prob = RuleSet.return_prob(rule[0],rule[1])
            #print "rule_prob is ",rule_prob
        #else:
        #    print 'not got rule for ',rule[1]
        c_prob = c_prob*rule_prob
        c_prob = c_prob*catProbFromGrammar(sc.funct,RuleSet)
    elif sc.direction=="back":
        rule = (sc.funct.toString(),sc.arg.toString()+'#####'+sc.toString())
        if RuleSet.check_target(rule[1]):
            rule_prob = RuleSet.return_prob(rule[0],rule[1])
            #print "rule_prob is ",rule_prob
        #else:
        #    print 'not got rule for ',rule[1]
        c_prob = c_prob*rule_prob
        c_prob = c_prob*catProbFromGrammar(sc.funct,RuleSet)

    return c_prob
    
# 2. get P(lf_{type}|cat) from lexicon.
# code in lexicon class
def getSemTypeProb(lexicon,sem_store,cat,lfType,posType,arity,varorder):
    #lexicon,sem_store,cat,lfType):
    # this should actually be, actual type, pos tag and arity
    #print "called it"    
    catkey = cat.toString()
    if lexicon.syntax.has_key(catkey):
        semTypeProbs = lexicon.get_sem_from_type_prob(sem_store,catkey,lfType,posType,arity,varorder)
                
        vo = ""
        for v in varorder:
            #print "v is ",v
            vo = vo+str(v)
            dictkey = (lfType,posType,arity,vo)

        if semTypeProbs.has_key(dictkey):
            return semTypeProbs[dictkey]
        #else:
            #print "not got lfType key ",lfType.toString()
        #print semTypeProbs
    else:
        #print "syntax not got key ",catkey
        return 0.0
        pass
        
    return 0.0 


# 3. get P(cat|lf_{type},postype).
def getPCatGivenLFType(cat,lfType,posType,arity,varorder,lexicon,sem_store,RuleSet,outputFile):

    
    pC = catProbFromGrammar(cat,RuleSet)
#    print "pC for ",cat.toString()," is ",pC
    # match lf_type
    # match pos_type
    
    # we care about this as it gives us the lambda orders
    # also it gives us
    # catstring

    pLgivenC = getSemTypeProb(lexicon,sem_store,cat,lfType,posType,arity,varorder)
    print >> outputFile,pC,":",pLgivenC,"  ",       
    #print "pLgivenC for ",cat.toString()," = ",pLgivenC
    pCgivenL = pLgivenC*pC # /pL 
    # this last term might be a little important - nah, only care about directional
    # prob
    return pCgivenL

# 4. get all probs to be compared
def outputCatProbs(posType,lfType,arity,cats,lexicon,sem_store,RuleSet,outputFile):
    norm = 0.0
    probList = []
    
    #semtype 
    for (cat,varorder) in cats:
        arity = len(varorder)
        #semtype = cat.getType().toString()
        pCgivenL = getPCatGivenLFType(cat,lfType,posType,arity,varorder,lexicon,sem_store,RuleSet,outputFile)
        norm += pCgivenL
        probList.append(pCgivenL)
    print >> outputFile,""
#    print "norm is ",norm," for ",posType
    for p in probList:
        #print "p is ",p
        if norm==0.0: print >> outputFile,1.0/len(probList),' ',
        else: print >> outputFile,p/norm,' ',
    
    print >> outputFile,''
