## ccg category class
# import exp
# import types

import pdb
from semType import *
from exp import *
import expFunctions
import re

trans_binding1 = re.compile('lambda\\ \\$0\\_\\{e\\}\\.lambda\\ \\$1\\_\\{e\\}\\.lambda\\ ' +
                            '\\$2\\_\\{r\\}\\.v\\|[^\\(\\)]+\\(\\$0\\,\\$1\\,\\$2\\)')
trans_binding2 = re.compile('lambda\\ \\$0\\_\\{e\\}\\.lambda\\ \\$1\\_\\{e\\}\\.lambda\\ ' +
                            '\\$2\\_\\{r\\}\\.v\\|[^()]+\\(\\$1\\,\\$0\\,\\$2\\)')


class synCat:
    def __init__(self, head, arg, direction):
        self.funct = head
        self.arg = arg
        # direction = "fwd" or "back"
        self.direction = direction

    def copy(self):
        return synCat(self.funct.copy(), self.arg.copy(), self.direction)

    def getType(self):
        return semType(self.arg.getType(), self.funct.getType())

    def prior(self):
        return self.funct.prior() * self.arg.prior()

    def equals(self, other):
        if other.atomic() != self.atomic():
            return False
        retval = self.funct.equals(other.funct) and \
                 self.arg.equals(other.arg) and \
                 self.direction == other.direction
        return retval

    def atomic(self):
        return False

    def arity(self):
        if self.atomic(): return 1
        return 1 + self.funct.arity()

    @staticmethod
    def allSynCatsWithPos(e):  # put subscripts and shit
        if e.__class__ == variable:
            return synCat.allSynCats(e.type())
        catType = e.type()
        synCats = []
        ## just function application for now
        if catType.atomic():
            if catType.isE(): return [synCat.np]
            if catType.isT(): return [synCat.st]
            if catType.isEvent(): return []

        othercats = getCatAug(e)
        if othercats:
            return othercats

        # don't we want to steal quite a lot from the parent?
        # do for now and come back to it....
        argCats = synCat.allSynCatsWithPos(e.getVar())
        functCats = synCat.allSynCatsWithPos(e.getFunct())
        for argCat in argCats:
            # bracketing, but need to be careful about where it goes.
            # should put in implicit bracketing. 
            # arg not always bracketed
            for functCat in functCats:
                # fwd
                synCats.append(synCat(functCat, argCat, "fwd"))
                # back
                synCats.append(synCat(functCat, argCat, "back"))

        return synCats

    @staticmethod
    def allSynCats(catType):
        synCats = []
        # just function application for now
        if catType.atomic():
            if catType.isE(): return [synCat.np]
            if catType.isT(): return [synCat.st]
            if catType.isEvent(): return []

        othercats = getCat(catType)
        if othercats:
            return othercats

        # don't we want to steal quite a lot from the parent?
        # do for now and come back toit....
        argCats = synCat.allSynCats(catType.getArg())
        functCats = synCat.allSynCats(catType.getFunct())
        for argCat in argCats:
            # bracketing, but need to be careful about where it goes.
            # should put in implicit bracketing. 
            # arg not always bracketed 

            # print "argCat is ",argCat.toString()
            for functCat in functCats:
                # print "functCat is ",functCat.toString()
                # fwd
                synCats.append(synCat(functCat, argCat, "fwd"))
                # back
                synCats.append(synCat(functCat, argCat, "back"))

        return synCats

    def toString(self):
        catString = "(" + self.funct.toString()
        if self.direction == "fwd":
            catString += "/"
        elif self.direction == "back":
            catString = catString + "\\"
        catString = catString + self.arg.toString()
        catString = catString + ")"
        return catString

    @staticmethod
    def readCat(synstring):
        # print "synstring is ",synstring
        if synstring[0] != "(":
            # atomic
            # if synstring == "PP":
            #     return synCat.pp
            if synstring == "NP":
                return synCat.np
            elif synstring == "S":
                return synCat.s
            elif synstring == "N":
                return synCat.n
            elif synstring == "Swh":
                return synCat.swh
            elif synstring == "Syn":
                return synCat.q
            elif synstring == "St":
                return synCat.st
        else:
            synstring = synstring[1:-1]
            numbrack = 0
            i = 0
            foundslash = False
            while not foundslash:
                # print "i is ",i," char is ",synstring[i]," numbrack is ",numbrack
                if synstring[i] in ["/", "\\"] and numbrack == 0:
                    funct = synCat.readCat(synstring[:i])
                    arg = synCat.readCat(synstring[i + 1:])
                    if synstring[i] == "/": dir = "fwd"
                    if synstring[i] == "\\": dir = "back"
                    foundslash = True
                    return synCat(funct, arg, dir)
                elif synstring[i] == "(":
                    numbrack += 1
                elif synstring[i] == ")":
                    numbrack -= 1
                i += 1


class npCat(synCat):
    def __init__(self):
        self.head = "NP"

    def atomic(self):
        return True

    def prior(self):
        return 0.2

    def getType(self):
        return semType.e

    @staticmethod
    def getStaticType():
        return semType.e

    def copy(self):
        return self

    def toString(self):
        return self.head

    def equals(self, other):
        return other.__class__ == npCat


class nCat(synCat):
    def __init__(self):
        self.head = "N"

    def atomic(self):
        return True

    def prior(self):
        return 0.2

    def getType(self):
        return semType(semType.e, semType.t)

    @staticmethod
    def getStaticType():
        return semType(semType.e, semType.t)

    def copy(self):
        return self

    def toString(self):
        return self.head

    def equals(self, other):
        return other.__class__ == nCat


class stCat(synCat):
    def __init__(self):
        self.head = "St"

    def atomic(self):
        return True

    def prior(self):
        return 0.2

    def getType(self):
        return semType.t

    @staticmethod
    def getStaticType():
        return semType.t

    def copy(self):
        return self

    def toString(self):
        return self.head

    def equals(self, other):
        return other.__class__ == stCat


class sCat(synCat):
    def __init__(self):
        self.head = "S"

    def atomic(self):
        return True

    def prior(self):
        return 0.2

    def getType(self):
        return semType(semType.event, semType.t)

    @staticmethod
    def getStaticType():
        return semType(semType.event, semType.t)

    def copy(self):
        return self

    def toString(self):
        return self.head

    def equals(self, other):
        return other.__class__ == sCat


class sWhCat(synCat):
    def __init__(self):
        self.head = "Swh"

    def atomic(self):
        return True

    def prior(self):
        return 0.2

    def getType(self):
        return semType(semType.e, semType(semType.event, semType.t))

    @staticmethod
    def getStaticType():
        return semType(semType.e, semType(semType.event, semType.t))

    def copy(self):
        return self

    def toString(self):
        return self.head

    def equals(self, other):
        return other.__class__ == sWhCat


class qCat(synCat):
    def __init__(self):
        self.head = "Syn"

    def atomic(self):
        return True

    def prior(self):
        return 0.2

    def getType(self):
        return semType(semType.event, semType.t)

    @staticmethod
    def getStaticType():
        return semType.t

    def copy(self):
        return self

    def toString(self):
        return self.head

    def equals(self, other):
        return other.__class__ == qCat


def getCat(catType):
    cats = []
    # <e,t>
    if catType.equals(nCat.getStaticType()):
        cats.append(synCat.n)
    # <ev,t>
    if catType.equals(sCat.getStaticType()):
        cats.append(synCat.s)
    return cats


def getCatAug(e):
    t = e.type()
    cats = []
    if t.equals(nCat.getStaticType()):
        cats.append(synCat.n)
        return cats
    if t.equals(sCat.getStaticType()):
        cats.append(synCat.s)
    return cats


class cat:
    def __init__(self, syn, sem):
        self.syn = syn
        self.sem = sem

    def prior(self):
        return self.syn.prior()

    def copy(self):
        return cat(self.syn.copy(), self.sem.copy())

    def equals(self, other):
        return self.syn.equals(other.syn) and self.sem.equals(other.sem)

    def apply(self, c, dir):
        if self.syn.atomic(): return None
        newcat = None
        if self.syn.arg.equals(c.syn) and self.syn.direction == dir:
            retsem = self.sem.apply(c.sem)
            if retsem:
                newcat = cat(self.syn.funct.copy(), retsem)
        else:
            return None
        return newcat

    def compose(self, c, dir):
        if self.syn.atomic() or c.syn.atomic(): return None
        if self.syn.direction != dir or c.syn.direction != dir: return None
        newcat = None
        syntomatch = c.syn
        argsremoved = []
        retsem = None
        # crossing???
        # return None
        while syntomatch:
            # something about number of arguments
            if self.syn.arg.equals(syntomatch) and self.syn.direction == dir:
                retsem = self.sem.compose(c.sem)
                break
            elif not syntomatch.atomic() and syntomatch.direction == dir:
                argsremoved.append(syntomatch.arg)
                syntomatch = syntomatch.funct

            else:
                syntomatch = None
        # how does the compcat work???
        newsyn = self.syn.funct.copy()
        for a in reversed(argsremoved): newsyn = synCat(newsyn, a, dir)

        if retsem:
            newcat = cat(newsyn, retsem)
        else:
            print "compfuckup"
        return newcat

    def combine(self, c):
        pass

    def getSem(self):
        return self.sem

    def getSyn(self):
        return self.syn

    def synString(self):
        return self.syn.toString()

    def semString(self):
        return self.sem.toString(True)

    def allPairs(self, catStore):
        if catStore.has_key(self.toString()): return catStore[self.toString()]
        pairs = []

        # really want to know :
        #    a) which cats can actually be played with
        #    b) whether or not we're going for composition

        # newLambdas is the number of new lambda terms in the argument
        # numByComp says how many lambda terms composition was used on
        sem_pairs = self.sem.makePairs()
        for (parentSem, childSem, numNew, numByComp, fixeddircats) in sem_pairs:

            # fixed dir cats is a funny one since many of the cats have
            # a lambda associated with them but this is fine since

            # print "here"
            # print "\nexp split is ",parentSem.toString(True)," ",childSem.toString(True)
            # print parentSem.toString(True)
            # print childSem.toString(True)
            # don't really want to do al
            # pairs.ap

            # ps = parentSem.copy()
            # cs = childSem.copy()

            # nr = ps.apply(cs)

            # if nr:
            # if not nr.equals(self.sem): print parentSem.toString(True)," applied to ",childSem.toString(True),\
            # " does not get us back to ",self.sem.toString(True)," gets us to ",nr.toString(True)
            # else: print "matches woooh"
            # else:
            # nr = ps.compose(cs)
            # if not nr or not nr.equals(self.sem): print "does not match under comp",\
            # parentSem.toString(True)," composed with to ",childSem.toString(True),\
            # " does not get us back to ",self.sem.toString(True)," gets us to ",nr.toString(True)
            # else: print "compmatches ",parentSem.toString(True),childSem.toString(True),nr.toString(True)

            # only interested in the most significant child lambdas that
            # really have to go

            # want the child to steal (borrow) a lot from the parent

            # all nulls by composition
            isnull = False
            if fixeddircats is None:
                functcat = self.syn.copy()

                argcat = self.syn.copy()
                childcat = self.syn.copy()
                if self.syn != synCat.swh:
                    csf = synCat(functcat, argcat, "fwd")
                    cfp = cat(csf, parentSem)
                    cfc = cat(childcat, childSem)
                    append_pairs(pairs, (cfp, cfc, "fwd", numByComp))

                    functcat = self.syn.copy()
                    argcat = self.syn.copy()
                    csb = synCat(functcat, argcat, "back")
                    cbp = cat(csb, parentSem)
                    cbc = cat(childcat, childSem)
                    append_pairs(pairs, (cbc, cbp, "back", numByComp))
                    continue
                else:
                    for sc in synCat.allSynCatsWithPos(self.sem):
                        argcat = sc.copy()
                        childcat = sc.copy()
                        csf = synCat(functcat, argcat, "fwd")
                        cfp = cat(csf, parentSem)
                        cfc = cat(childcat, childSem)
                        append_pairs(pairs, (cfp, cfc, "fwd", numByComp))

                        functcat = self.syn.copy()
                        csb = synCat(functcat, argcat, "back")
                        cbp = cat(csb, parentSem)
                        cbc = cat(childcat, childSem)
                        append_pairs(pairs, (cbc, cbp, "back", numByComp))
                    continue

            compargs = []
            lterm = self.sem
            # check that type and direction match
            # break if not 
            compfunct = self.syn
            canDoComp = False
            for i in range(numByComp):
                compvartype = lterm.var.type()
                if compfunct.atomic():
                    canDoComp = False
                    break
                compcat = compfunct.arg
                compfunct = compfunct.funct
                if canDoComp and canDoComp != self.syn.direction:
                    canDoComp = False
                    break
                elif not canDoComp:
                    canDoComp = self.syn.direction

                compcattype = compcat.getType()
                if not compvartype.equals(compcattype): canDoComp = False
                lterm = lterm.funct
                # need to make sure that type matches if allowed
                compargs.append(compcat)

            if canDoComp:
                pass
            else:
                numByComp = 0
            returncat = self.syn
            currcat = self.syn
            currlam = self.sem
            catstoappend = []
            # want to know about arg cat 
            lvars = childSem.getLvars()
            t = childSem.type()
            pt = childSem
            for i in range(numByComp):
                if currcat.atomic():
                    break
                del lvars[i]

                t = t.functType
                pt = pt.getFunct()
                catstoappend.append((currcat.arg, currcat.direction))
                currcat = currcat.funct
                currlam = currlam.funct

            compbase = currcat
            # is this ok??? yes???
            # do lots of debugging!!!!
            # get ALL splits of a top cat
            istyperaised = False
            if len(fixeddircats) > 0 and fixeddircats[0] == "typeraised":
                fixeddircats = fixeddircats[1:]
                istyperaised = True
                t = parentSem.getVar().getArg(0).type()
                pt = parentSem.getVar().getArg(0)

            # want cat for root and outermost args

            # this is going to be important
            # catstoappend = []
            # currcat = self.syn

            # this is super wrong. we need to know which cats 
            # they are.

            # now we are interested in the cats for things that come 
            # from seen lambdas
            fixedcats = []
            seenfixeddircats = 0

            while seenfixeddircats < min(len(fixeddircats), currcat.arity() - 1):
                if numByComp > seenfixeddircats:
                    seenfixeddircats += 1
                else:
                    if currlam.var == fixeddircats[seenfixeddircats]:
                        seenfixeddircats += 1
                        if currcat.atomic(): break
                        fixedcats.append((currcat.arg, currcat.direction))
                        t = t.functType
                        pt = pt.getFunct()
                    currcat = currcat.funct
                    currlam = currlam.funct

            # here there could be some dynamic programming surely???
            # need to work out the sharing for non comp cats
            for sc in synCat.allSynCatsWithPos(pt):
                # fixed because lambda terms equivalent 
                # to above
                for ca in reversed(fixedcats):
                    sc = synCat(sc, ca[0], ca[1])

                # WHICH CATS ARE SHARED????
                # inside ones

                pscf = synCat(compbase, sc, "fwd")
                pscb = synCat(compbase, sc, "back")

                # fixed by composition
                for ca in reversed(catstoappend):
                    sc = synCat(sc, ca[0], ca[1])
                # need to rebuild directional child cat
                childCat = cat(sc, childSem)

                if not canDoComp or canDoComp == "fwd":
                    if istyperaised: print "typeraised"
                    parentCat = cat(pscf, parentSem)
                    if not pscf.getType().equals(parentSem.type()):
                        print "types dont match 2 : ", parentCat.toString(), \
                            " ", pscf.getType().toString(), " ", parentSem.type().toString(), \
                            " comp is ", canDoComp, "\n"
                        print parentCat.toString(), childCat.toString()
                        print "NOT ADDING THIS"
                    elif not istyperaised:
                        append_pairs(pairs, (parentCat, childCat, "fwd", numByComp))
                    else:
                        if canDoComp: raise (StandardError("typeraise on comp"))
                        # want the parent to be looking in the opposite way from the
                        # child
                        parentCat = cat(pscf, childSem)
                        typeRaisedChildSyn = synCat(returncat, pscf, "back")
                        typeRaisedChild = cat(typeRaisedChildSyn, parentSem)
                        append_pairs(pairs, (parentCat, typeRaisedChild, "back", numByComp))
                        childCat = parentCat
                        parentCat = typeRaisedChild

                if not canDoComp or canDoComp == "back":
                    if istyperaised: print "typeraised"
                    parentCat = cat(pscb, parentSem)
                    if not pscb.getType().equals(parentSem.type()):
                        print "types dont match 3 : ", parentCat.toString(), \
                            " ", pscb.getType().toString(), " ", parentSem.type().toString(), \
                            " comp is ", canDoComp, "\n"
                        print "NOT ADDING THIS"
                    elif not istyperaised:
                        append_pairs(pairs, (childCat, parentCat, "back", numByComp))
                    else:
                        if canDoComp: raise (StandardError("typeraise on comp"))
                        parentCat = cat(pscb, childSem)
                        typeRaisedChildSyn = synCat(returncat, pscb, "fwd")
                        typeRaisedChild = cat(typeRaisedChildSyn, parentSem)
                        append_pairs(pairs, (typeRaisedChild, parentCat, "fwd", numByComp))
                        childCat = parentCat
                        parentCat = typeRaisedChild

                if not canDoComp:
                    pc = parentCat.copy()
                    cc = childCat.copy()
                    if istyperaised:
                        nc = pc.apply(cc, "fwd")
                    else:
                        nc = pc.apply(cc, "back")
                    if nc:
                        if not nc.equals(self):
                            # print "not back to orig, should be ", self.toString()
                            pass
                    else:
                        print "got back to orig "
                else:
                    pc = parentCat.copy()
                    cc = childCat.copy()
                    nc = pc.compose(cc, canDoComp)
                    if nc:
                        if not nc.equals(self):
                            # print "not back to orig, should be ", self.toString()
                            pass

        catStore[self.toString()] = pairs
        return pairs

    def toString(self):
        return self.syn.toString() + ":" + self.sem.toString(True)

    @staticmethod
    def readCat(catstring):
        synstring = catstring.split(" :: ")[0]
        semstring = catstring.split(" :: ")[1]
        (semrep, expString) = expFunctions.makeExpWithArgs(semstring, {})
        syncat = synCat.readCat(synstring)
        c = cat(syncat, semrep)
        return c


###########################
# CODE OMRI ADDED 
###########################

def append_pairs(pairs, new_entry):
    if check_restrictions(new_entry[0]) and check_restrictions(new_entry[1]):
        pairs.append(new_entry)


def all_directions(syn_cat):
    """
    Returns a list of all of the directions of the arguments of the category.
    """
    output = []
    c = syn_cat
    while not c.atomic():
        output.append(c.direction)
        c = c.funct
    return output


def check_restrictions(cur_cat):
    """
    Receives an instance of cat.cat.
    Returns True iff the entry does not violate the binding restrictions, i.e., that
    the arguments should appear in the same order in the logical 
    form and in the syntactic category, whenever you have a case of opposite directions of slashes.
    """
    directions = all_directions(cur_cat.getSyn())
    if len(directions) >= 2 and (directions[-2:] == ['back', 'fwd'] or directions[-2:] == ['fwd', 'back']):
        order_var = sorted(orderOfVariables(cur_cat), key=lambda x: x[0])
        order_var = order_var[
                    :len(directions)]  # we don't care about lambdas that belong to the atomic category of the functor
        if len(order_var) < 2:
            return True
        if any([len(x) < 2 for x in order_var]):
            print('VIOLATION: vacuous variable found in ' + cur_cat.synString() + ' ' + cur_cat.semString())
            return False
        rightOrder = (order_var[-1][1] > order_var[-2][1])
        if rightOrder and (directions[-2:] == ['back', 'fwd'] or \
                                       directions[-2:] == ['fwd', 'back']):
            return False
    return True


def orderOfVariables(cur_cat):
    """
    In a string s, returns the order of expressions of the form $[0-9]
    Returns a list of the variables in the formula ordered by the order of their
    first occurrance.
    """
    all_vars = re.findall('\\$[0-9]', cur_cat.semString())
    D = {}
    for ind, v in enumerate(all_vars):
        cur = D.get(v, [])
        cur.append(ind)
        D[v] = cur
    return D.values()



# make static cats #
synCat.np = npCat()
synCat.s = sCat()
synCat.st = stCat()
synCat.swh = sWhCat()
synCat.q = qCat()
synCat.n = nCat()


def main(argv=None):
    # toprepstring = ""
    ##toprepstring = "lambda $0_{<e,t>}.lambda $1_{e}.lambda $2_{ev}.aux|will&COND(v|like($1,qn|more($3,and($0($3),n|juice($3))),$2))"
    ##toprepstring = "lambda $0_{e}.and(n|grape($0),n|juice($0))"
    ##toprepstring = "lambda $0_{ev}.v|go(pro|you,$0)"
    # toprepstring = "not(adj|sure(pro|I))"
    ##lambda $0_{<<<<e,t>,<e,t>>,<<e,t>,<ev,t>>>,<<<e,t>,<e,t>>,<<e,t>,<ev,t>>>>}.lambda $1_{<<<e,t>,<e,t>>,<<e,t>,<ev,t>>>}.lambda $2_{<<e,t>,<e,t>>}.lambda $3_{<e,t>}.Q($0($1,$2,$3))"
    # sem = exp.makeExpWithArgs(toprepstring,{})[0]
    ##Q(eq(pro|it,det|a($0,and(adj|loud($0),n|boom($0)))))",{})[0]
    # sc = synCat.allSynCats(sem.type())[0]
    # topCat = cat(sc,sem)
    # c = topCat
    exp.allowTypeRaising = True

    # catstring = "((Syn\(S/S))\(((S/S)/(S/S))/S)) :: lambda $0_{<<ev,t>,<<<ev,t>,t>,<<ev,t>,t>>>}.$0(lambda $1_{ev}.aux|will&COND(v|like(pro|you,qn|more($2,and(n|grape($2),n|juice($2))),$1)))"
    catstring = "S :: not(adj|sure(pro|I))"
    catstring = "S :: lambda $0_{ev}.and(v|go&PAST(n:prop|Momma,$0),prep|to(n:prop|Boston,$0))"
    catstring = "S :: lambda $0_{ev}.and(v|put&ZERO(pro|you,pro|them,$0),prep|on(det|the($1,n|table($1)),$0))"

    # (S\NP) :: lambda $0_{e}.not(adj|sure($0))"
    c = cat.readCat(catstring)
    if c: print "cat is : ", c.toString()
    pairs = c.allPairs({})
    print len(pairs), " splits"
    print "splits are :: "
    for pair in pairs:
        print pair[0].toString(), " ", pair[1].toString()
    print "\n"

    # catstring = "(((Syn\(S/S))\(((S/S)/(S/S))/S))/S) :: lambda $0_{<ev,t>}.lambda $1_{<<ev,t>,<<<ev,t>,t>,<<ev,t>,t>>>}.$1($0)"
    # c = cat.readCat(catstring)
    # if c :
    # print "cat is : ",c.toString()
    # print "sem type is : ",c.sem.type().toString()
    ##print "splits are :: "
    ##for pair in c.allPairs({}):
    ##print pair[0].toString()," ",pair[1].toString()
    # print "\n"

    # catstring = "(S/NP) :: lambda $0_{e}.lambda $1_{ev}.v|eat($0,pro|it,$1)"
    # c = cat.readCat(catstring)
    # if c : print "cat is : ",c.toString()
    # print "splits are :: "
    # for pair in c.allPairs({}):
    # print pair[0].toString()," ",pair[1].toString()
    # print "\n"

    # catstring = "(S\\NP) :: lambda $0_{e}.lambda $1_{ev}.v|have($0,det|a($2,n|hat($2)),$1)"
    # c = cat.readCat(catstring)
    # if c : print "cat is : ",c.toString()
    # print "splits are :: "
    # for pair in c.allPairs({}):
    # print pair[0].toString()," ",pair[1].toString()
    # print "\n"


if __name__ == "__main__":
    main()
