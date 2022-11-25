from exp import Variable, make_exp_with_args
from sem_type import SemType
import re


class Cat:
    def __init__(self, syn, sem):
        self.syn = syn
        self.sem = sem

    def prior(self):
        return self.syn.prior()

    def copy(self):
        return Cat(self.syn.copy(), self.sem.copy())

    def equals(self, other):
        return self.syn.equals(other.syn) and self.sem.equals(other.sem)

    def apply(self, c, dir):
        if self.syn.atomic(): return None
        newcat = None
        if self.syn.arg.equals(c.syn) and self.syn.direction == dir:
            retsem = self.sem.apply(c.sem)
            if retsem:
                newcat = Cat(self.syn.funct.copy(), retsem)
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
        while syntomatch:
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
        for a in reversed(argsremoved): newsyn = SynCat(newsyn, a, dir)

        if retsem:
            newcat = Cat(newsyn, retsem)
        else:
            #print("compfuckup")
            pass
        return newcat

    def combine(self, c):
        pass

    def get_sem(self):
        return self.sem

    def get_syn(self):
        return self.syn

    def syn_string(self):
        return self.syn.to_string()

    def sem_string(self):
        return self.sem.to_string(True)

    def all_pairs(self, cat_store):
        if self.to_string() in cat_store: return cat_store[self.to_string()]
        pairs = []

        # really want to know :
        #    a) which cats can actually be played with
        #    b) whether or not we're going for composition

        # new_lambdas is the number of new lambda terms in the argument
        # num_by_comp says how many lambda terms composition was used on
        sem_pairs = self.sem.make_pairs()
        for (parent_sem, child_sem, num_new, num_by_comp, fixeddircats) in sem_pairs:
            # want the child to steal (borrow) a lot from the parent
            if fixeddircats is None:
                functcat = self.syn.copy()
                argcat = self.syn.copy()
                childcat = self.syn.copy()
                if self.syn != SynCat.swh:
                    csf = SynCat(functcat, argcat, "fwd")
                    cfp = Cat(csf, parent_sem)
                    cfc = Cat(childcat, child_sem)
                    append_pairs(pairs, (cfp, cfc, "fwd", num_by_comp))

                    functcat = self.syn.copy()
                    argcat = self.syn.copy()
                    csb = SynCat(functcat, argcat, "back")
                    cbp = Cat(csb, parent_sem)
                    cbc = Cat(childcat, child_sem)
                    append_pairs(pairs, (cbc, cbp, "back", num_by_comp))
                    continue

                else:
                    for sc in SynCat.all_syn_cats_with_pos(self.sem):
                        argcat = sc.copy()
                        childcat = sc.copy()
                        csf = SynCat(functcat, argcat, "fwd")
                        cfp = Cat(csf, parent_sem)
                        cfc = Cat(childcat, child_sem)
                        append_pairs(pairs, (cfp, cfc, "fwd", num_by_comp))

                        functcat = self.syn.copy()
                        csb = SynCat(functcat, argcat, "back")
                        cbp = Cat(csb, parent_sem)
                        cbc = Cat(childcat, child_sem)
                        append_pairs(pairs, (cbc, cbp, "back", num_by_comp))
                    continue

            compargs = []
            lterm = self.sem
            # check that type and direction match
            # break if not
            compfunct = self.syn
            can_do_comp = False
            for i in range(num_by_comp):
                compvartype = lterm.var.type()
                if compfunct.atomic():
                    can_do_comp = False
                    break
                compcat = compfunct.arg
                compfunct = compfunct.funct
                if can_do_comp and can_do_comp != self.syn.direction:
                    can_do_comp = False
                    break
                elif not can_do_comp:
                    can_do_comp = self.syn.direction

                compcattype = compcat.get_type()
                if not compvartype.equals(compcattype): can_do_comp = False
                lterm = lterm.funct
                # need to make sure that type matches if allowed
                compargs.append(compcat)

            if can_do_comp:
                pass
            else:
                num_by_comp = 0
            returncat = self.syn
            currcat = self.syn
            currlam = self.sem
            catstoappend = []
            # want to know about arg Cat
            lvars = child_sem.get_lvars()
            t = child_sem.type()
            pt = child_sem
            for i in range(num_by_comp):
                if currcat.atomic():
                    break
                del lvars[i]

                t = t.funct_type
                pt = pt.get_funct()
                catstoappend.append((currcat.arg, currcat.direction))
                currcat = currcat.funct
                currlam = currlam.funct

            compbase = currcat
            # get ALL splits of a top Cat
            istyperaised = False
            if len(fixeddircats) > 0 and fixeddircats[0] == "typeraised":
                fixeddircats = fixeddircats[1:]
                istyperaised = True
                t = parent_sem.get_var().get_arg(0).type()
                pt = parent_sem.get_var().get_arg(0)

            # want Cat for root and outermost args

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
                if num_by_comp > seenfixeddircats:
                    seenfixeddircats += 1
                else:
                    if currlam.var == fixeddircats[seenfixeddircats]:
                        seenfixeddircats += 1
                        if currcat.atomic(): break
                        fixedcats.append((currcat.arg, currcat.direction))
                        t = t.funct_type
                        pt = pt.get_funct()
                    currcat = currcat.funct
                    currlam = currlam.funct

            # here there could be some dynamic programming surely???
            # need to work out the sharing for non comp cats
            for sc in SynCat.all_syn_cats_with_pos(pt):
                # fixed because lambda terms equivalent
                # to above
                for ca in reversed(fixedcats):
                    sc = SynCat(sc, ca[0], ca[1])

                # WHICH CATS ARE SHARED????
                # inside ones

                pscf = SynCat(compbase, sc, "fwd")
                pscb = SynCat(compbase, sc, "back")

                # fixed by composition
                for ca in reversed(catstoappend):
                    sc = SynCat(sc, ca[0], ca[1])
                # need to rebuild directional child Cat
                child_cat = Cat(sc, child_sem)

                if not can_do_comp or can_do_comp == "fwd":
                    if istyperaised: print("typeraised")
                    parent_cat = Cat(pscf, parent_sem)
                    if not pscf.get_type().equals(parent_sem.type()):
                        #print("types dont match 2 : ", parent_cat.to_string(), \
                        #    " ", pscf.get_type().to_string(), " ", parent_sem.type().to_string(), \
                        #    " comp is ", can_do_comp, "\n")
                        #print(parent_cat.to_string(), child_cat.to_string())
                        #print("NOT ADDING THIS")
                        pass
                    elif not istyperaised:
                        append_pairs(pairs, (parent_cat, child_cat, "fwd", num_by_comp))
                    else:
                        if can_do_comp: raise Exception
                        # want the parent to be looking in the opposite way from the
                        # child
                        parent_cat = Cat(pscf, child_sem)
                        type_raised_child_syn = SynCat(returncat, pscf, "back")
                        type_raised_child = Cat(type_raised_child_syn, parent_sem)
                        append_pairs(pairs, (parent_cat, type_raised_child, "back", num_by_comp))
                        child_cat = parent_cat
                        parent_cat = type_raised_child

                if not can_do_comp or can_do_comp == "back":
                    if istyperaised: print("typeraised")
                    parent_cat = Cat(pscb, parent_sem)
                    if not pscb.get_type().equals(parent_sem.type()):
                        #print("types dont match 3 : ", parent_cat.to_string(), \
                        #    " ", pscb.get_type().to_string(), " ", parent_sem.type().to_string(), \
                        #    " comp is ", can_do_comp, "\n")
                        #print("NOT ADDING THIS")
                        pass
                    elif not istyperaised:
                        append_pairs(pairs, (child_cat, parent_cat, "back", num_by_comp))
                    else:
                        if can_do_comp: raise Exception
                        parent_cat = Cat(pscb, child_sem)
                        type_raised_child_syn = SynCat(returncat, pscb, "fwd")
                        type_raised_child = Cat(type_raised_child_syn, parent_sem)
                        append_pairs(pairs, (type_raised_child, parent_cat, "fwd", num_by_comp))
                        child_cat = parent_cat
                        parent_cat = type_raised_child

                if not can_do_comp:
                    pc = parent_cat.copy()
                    cc = child_cat.copy()
                    if istyperaised:
                        nc = pc.apply(cc, "fwd")
                    else:
                        nc = pc.apply(cc, "back")
                    if nc:
                        if not nc.equals(self):
                            # print "not back to orig, should be ", self.to_string()
                            pass
                    else:
                        #print("got back to orig ")
                        pass
                else:
                    pc = parent_cat.copy()
                    cc = child_cat.copy()
                    nc = pc.compose(cc, can_do_comp)
                    if nc:
                        if not nc.equals(self):
                            # print "not back to orig, should be ", self.to_string()
                            pass

        cat_store[self.to_string()] = pairs
        return pairs

    def to_string(self):
        return self.syn.to_string() + ":" + self.sem.to_string(True)

    @staticmethod
    def read_cat(catstring):
        synstring = catstring.split(" :: ")[0]
        semstring = catstring.split(" :: ")[1]
        (semrep, exp_string) = make_exp_with_args(semstring, {})
        syncat = SynCat.read_cat(synstring)
        c = Cat(syncat, semrep)
        return c

class SynCat:
    def __init__(self, head, arg, direction):
        self.funct = head
        self.arg = arg
        # direction = "fwd" or "back"
        self.direction = direction

    def copy(self):
        return SynCat(self.funct.copy(), self.arg.copy(), self.direction)

    def get_type(self):
        return SemType(self.arg.get_type(), self.funct.get_type())

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
    def all_syn_cats_with_pos(e):  # put subscripts and shit
        if e.__class__ == Variable:
            return SynCat.all_syn_cats(e.type())
        cat_type = e.type()
        SynCats = []
        # just function application for now
        if cat_type.atomic():
            if cat_type.is_e(): return [SynCat.np]
            if cat_type.is_t(): return [SynCat.st]
            if cat_type.is_event(): return []

        othercats = get_cat_aug(e)
        if othercats:
            return othercats

        # don't we want to steal quite a lot from the parent?
        # do for now and come back to it....
        arg_cats = SynCat.all_syn_cats_with_pos(e.get_var())
        funct_cats = SynCat.all_syn_cats_with_pos(e.get_funct())
        for arg_cat in arg_cats:
            # bracketing, but need to be careful about where it goes.
            # should put in implicit bracketing.
            # arg not always bracketed
            for funct_cat in funct_cats:
                # fwd
                SynCats.append(SynCat(funct_cat, arg_cat, "fwd"))
                # back
                SynCats.append(SynCat(funct_cat, arg_cat, "back"))

        return SynCats

    @staticmethod
    def all_syn_cats(cat_type):
        SynCats = []
        # just function application for now
        if cat_type.atomic():
            if cat_type.is_e(): return [SynCat.np]
            if cat_type.is_t(): return [SynCat.st]
            if cat_type.is_event(): return []

        othercats = get_cat(cat_type)
        if othercats:
            return othercats

        # don't we want to steal quite a lot from the parent?
        # do for now and come back toit....
        arg_cats = SynCat.all_syn_cats(cat_type.get_arg())
        funct_cats = SynCat.all_syn_cats(cat_type.get_funct())
        for arg_cat in arg_cats:
            # bracketing, but need to be careful about where it goes.
            # should put in implicit bracketing.
            # arg not always bracketed
            for funct_cat in funct_cats:
                # fwd
                SynCats.append(SynCat(funct_cat, arg_cat, "fwd"))
                # back
                SynCats.append(SynCat(funct_cat, arg_cat, "back"))

        return SynCats

    def to_string(self):
        cat_string = "(" + self.funct.to_string()
        if self.direction == "fwd":
            cat_string += "/"
        elif self.direction == "back":
            cat_string = cat_string + "\\"
        cat_string = cat_string + self.arg.to_string()
        cat_string = cat_string + ")"
        return cat_string

    @staticmethod
    def read_cat(synstring):
        if synstring[0] != "(":
            if synstring == "NP":
                return SynCat.np
            elif synstring == "S":
                return SynCat.s
            elif synstring == "N":
                return SynCat.n
            elif synstring == "Swh":
                return SynCat.swh
            elif synstring == "Syn":
                return SynCat.q
            elif synstring == "St":
                return SynCat.st
        else:
            synstring = synstring[1:-1]
            numbrack = 0
            i = 0
            foundslash = False
            while not foundslash:
                if synstring[i] in ["/", "\\"] and numbrack == 0:
                    funct = SynCat.read_cat(synstring[:i])
                    arg = SynCat.read_cat(synstring[i + 1:])
                    if synstring[i] == "/": dir = "fwd"
                    if synstring[i] == "\\": dir = "back"
                    foundslash = True
                    return SynCat(funct, arg, dir)
                elif synstring[i] == "(":
                    numbrack += 1
                elif synstring[i] == ")":
                    numbrack -= 1
                i += 1

class NPCat(SynCat):
    def __init__(self):
        self.head = "NP"

    def atomic(self):
        return True

    def prior(self):
        return 0.2

    def get_type(self):
        return SemType.e

    @staticmethod
    def get_static_type():
        return SemType.e

    def copy(self):
        return self

    def to_string(self):
        return self.head

    def equals(self, other):
        return other.__class__ == NPCat

class NCat(SynCat):
    def __init__(self):
        self.head = "N"

    def atomic(self):
        return True

    def prior(self):
        return 0.2

    def get_type(self):
        return SemType(SemType.e, SemType.t)

    @staticmethod
    def get_static_type():
        return SemType(SemType.e, SemType.t)

    def copy(self):
        return self

    def to_string(self):
        return self.head

    def equals(self, other):
        return other.__class__ == NCat

class STCat(SynCat):
    def __init__(self):
        self.head = "St"

    def atomic(self):
        return True

    def prior(self):
        return 0.2

    def get_type(self):
        return SemType.t

    @staticmethod
    def get_static_type():
        return SemType.t

    def copy(self):
        return self

    def to_string(self):
        return self.head

    def equals(self, other):
        return other.__class__ == STCat

class SCat(SynCat):
    def __init__(self):
        self.head = "S"

    def atomic(self):
        return True

    def prior(self):
        return 0.2

    def get_type(self):
        return SemType(SemType.event, SemType.t)

    @staticmethod
    def get_static_type():
        return SemType(SemType.event, SemType.t)

    def copy(self):
        return self

    def to_string(self):
        return self.head

    def equals(self, other):
        return other.__class__ == SCat

class SWhCat(SynCat):
    def __init__(self):
        self.head = "Swh"

    def atomic(self):
        return True

    def prior(self):
        return 0.2

    def get_type(self):
        return SemType(SemType.e, SemType(SemType.event, SemType.t))

    @staticmethod
    def get_static_type():
        return SemType(SemType.e, SemType(SemType.event, SemType.t))

    def copy(self):
        return self

    def to_string(self):
        return self.head

    def equals(self, other):
        return other.__class__ == SWhCat

class QCat(SynCat):
    def __init__(self):
        self.head = "Syn"

    def atomic(self):
        return True

    def prior(self):
        return 0.2

    def get_type(self):
        return SemType(SemType.event, SemType.t)

    @staticmethod
    def get_static_type():
        return SemType.t

    def copy(self):
        return self

    def to_string(self):
        return self.head

    def equals(self, other):
        return other.__class__ == QCat

def get_cat(cat_type):
    cats = []
    # <e,t>
    if cat_type.equals(NCat.get_static_type()):
        cats.append(SynCat.n)
    # <ev,t>
    if cat_type.equals(SCat.get_static_type()):
        cats.append(SynCat.s)
    return cats

def get_cat_aug(e):
    t = e.type()
    cats = []
    if t.equals(NCat.get_static_type()):
        cats.append(SynCat.n)
        return cats
    if t.equals(SCat.get_static_type()):
        cats.append(SynCat.s)
    return cats

# CODE OMRI ADDED
def append_pairs(pairs, new_entry):
    if check_restrictions(new_entry[0]) and check_restrictions(new_entry[1]):
        pairs.append(new_entry)

def all_directions(SynCat):
    """
    Returns a list of all of the directions of the arguments of the category.
    """
    output = []
    c = SynCat
    while not c.atomic():
        output.append(c.direction)
        c = c.funct
    return output

def check_restrictions(cur_cat):
    """Receives an instance of Cat.Cat.
    Returns True iff the entry does not violate the binding restrictions, i.e., that
    the arguments should appear in the same order in the logical
    form and in the syntactic category, whenever you have a case of opposite directions of slashes.
    """
    directions = all_directions(cur_cat.get_syn())
    if len(directions) >= 2 and (directions[-2:] == ['back', 'fwd'] or directions[-2:] == ['fwd', 'back']):
        order_var = sorted(order_of_variables(cur_cat), key=lambda x: x[0])
        order_var = order_var[
                    :len(directions)]  # we don't care about lambdas that belong to the atomic category of the functor
        if len(order_var) < 2:
            return True
        if any([len(x) < 2 for x in order_var]):
            print(('VIOLATION: vacuous variable found in ' + cur_cat.syn_string() + ' ' + cur_cat.sem_string()))
            return False
        right_order = (order_var[-1][1] > order_var[-2][1])
        if right_order and (directions[-2:] == ['back', 'fwd'] or
                                       directions[-2:] == ['fwd', 'back']):
            return False
    return True

def order_of_variables(cur_cat):
    """
    In a string s, returns the order of expressions of the form $[0-9]
    Returns a list of the variables in the formula ordered by the order of their
    first occurrance.
    """
    all_vars = re.findall('\\$[0-9]', cur_cat.sem_string())
    D = {}
    for ind, v in enumerate(all_vars):
        cur = D.get(v, [])
        cur.append(ind)
        D[v] = cur
    return list(D.values())


# make static cats #
SynCat.np = NPCat()
SynCat.s = SCat()
SynCat.st = STCat()
SynCat.swh = SWhCat()
SynCat.q = QCat()
SynCat.n = NCat()
