# expression, on which everything else is built
from sem_type import SemType
from inspect import signature
from tools import permutations
import re


class Exp:
    var_num = 0
    event_num = 0
    empty_num = 0
    allow_type_raise = False
    def __init__(self, name, num_args, arg_types, pos_type):
        self.onlyinout = None
        self.linked_var = None
        self.name = name
        self.num_args = num_args
        assert num_args==len(arg_types)
        self.arg_types = arg_types
        self.arguments = []
        self.parents = []
        for a_t in arg_types:
            self.arguments.append(EmptyExp())
        self.set_return_type()
        self.function_exp = self
        self.pos_type = pos_type
        self.arg_set = False
        self.is_verb=False
        self.is_null = False
        self.inout = None
        self.double_quant = False
        self.str_shell_prefix = 'placeholder_p'
        self.str_ubl_prefix = self.name.replace(':','#')

    def repair_binding(self, orig):
        for arg, orig_arg in zip(self.arguments, orig.arguments):
            arg.repair_binding(orig_arg)

    def is_q(self):
        return False

    def set_is_verb(self):
        self.is_verb = True

    def check_if_verb(self):
        return self.is_verb

    def is_conj_v(self):
        return False

    def check_if_wh(self):
        is_lambda = isinstance(self,LambdaExp)
        if is_lambda:
            has_e_var = self.get_var().type() == SemType.e
            funct = self.get_funct()
            funct_is_lambda = isinstance(funct,LambdaExp)
            if is_lambda and has_e_var and funct_is_lambda:
                return True
        else:
            return False

    # only lambdas should be allowed to apply and compose.
    def apply(self, e):
        return None

    def compose(self, e):
        return None

    def set_is_null(self):
        self.is_null=True

    def get_is_null(self):
        return self.is_null

    def is_entity(self):
        return False

    def equals(self,other):
        pass

    def equals_placeholder(self,other):
        return self.equals(other)

    def add_parent(self, e):
        if e not in self.parents:
            self.parents.append(e)

    def args_filled(self):
        for a in self.arguments:
            if a.is_empty(): return False
        return True

    def set_arg(self, position, argument):
        self.arguments.pop(position)
        self.arguments.insert(position, argument)
        if isinstance(argument, Exp):
            argument.add_parent(self)
            self.arg_set = True

    def get_arg(self, position):
        if position>len(self.arguments)-1: print("only got "+str(len(self.arguments))+" arguments")
        else: return self.arguments[position]

    def num_args(self):
        return len(self.arguments)

    def replace2(self, e1, e2):
        if self == e1:
            return e2
        self.arguments = [None if a is None else a.replace2(e1, e2) for a in self.arguments]
        #i=0
        #for a in self.arguments:
            #self.arguments.pop(i)
            #self.arguments.insert(i, a.replace2(e1, e2))
            #i+=1
        return self

    def make_variable(self, e):
        if e in self.arguments:
            var = Variable(e)
            self.arguments[self.arguments.index(e)] = var
            return var

    def is_empty(self):
        return False

    def get_name(self):
        return self.name

    def print_out(self, top, var_num):
        print(self.to_string(top))

    def _to_string(self, top, extra_format):
        s = ''
        for a in self.arguments:
            if isinstance(a, Exp): s=s+a.to_string(False,extra_format)
            if self.arguments.index(a)<self.num_args-1: s=s+","
        if len(self.arguments)>0: s='('+s+')'
        if top:
            Exp.var_num = 0
            Exp.event_num = 0
            Exp.empty_num = 0
        return s

    def to_string(self, top=True, extra_format=None):
        if extra_format is None:
            prefix = self.name
        elif extra_format == 'shell':
            prefix = self.str_shell_prefix
        elif extra_format == 'ubl':
            prefix = self.name.replace(':','#')

        return prefix + self._to_string(top, extra_format)

    def add_arg(self, arg):
        self.arguments.append(arg)
        pass

    def set_return_type(self):
        pass

    def get_return_type(self):
        return self.return_type

    def type(self):
        print("shouldnt be asking for type here")

    def get_pos_type(self):
        if self.pos_type: return self.pos_type
        return None

    def top_node(self):
        if len(self.parents)==0: return self
        top = None
        for p in self.parents:
            if not p.top_node(): return None
            if top and top!=p.top_node():
                print(top, "   ", p.top_node())
                return None
            top = p.top_node()
        return top

    def clear_parents(self):
        self.parents = []
        for a in self.arguments:
            a.clear_parents()

    def clear_parents2(self):
        self.parents = []

    def recalc_parents(self, top):
        if top:
            self.clear_parents()
        for a in self.arguments:
            a.add_parent(self)
            a.recalc_parents(False)

    def all_sub_exps(self):
        sub_exps = []
        sub_exps.append(self)
        for d in self.arguments:
            sub_exps.extend(d.all_sub_exps())
        return sub_exps

    def all_extractable_sub_exps(self):
        sub_exps = []
        sub_exps.append(self)
        for d in self.arguments:
            sub_exps.extend(d.all_extractable_sub_exps())
        return sub_exps

    def all_args(self):
        return self.arguments

    def get_all_vars(self, vars):
        for a in self.arguments:
            a.get_all_vars(vars)

    def vars_above(self, other, vars):
        if self==other: return
        for a in self.arguments:
            a.vars_above(other, vars)

    def unbound_vars(self):
        bound_vars = []
        vars = []
        sub_exps = self.all_sub_exps()
        for e in sub_exps:
            if isinstance(e,Variable) and e.binder:
                if e.binder in sub_exps:
                    bound_vars.append(e)
        self.get_all_vars(vars)
        unboundvars = []
        for v in vars:
            if v not in bound_vars and v!=self: unboundvars.append(v)
        return unboundvars

    def partition_vars(self, other):
        all_vars = []
        self.get_all_vars(all_vars)
        above_vars = []
        self.vars_above(other, above_vars)
        below_vars = []
        other.get_all_vars(below_vars)
        both_vars = []
        for v in all_vars:
            if v in below_vars:
                both_vars.append(v)
        return (below_vars, above_vars, both_vars)

    def pullout(self, e, vars, num_new_lam):
        """Return a pair copy for each way to pull the thing
         out. can be > 1 because of composition. Each pair needs
         to say how many lambda terms go with composition.
         just have a different definition in LambdaExp???
         """
        vargset = []
        for v in vars:
            vset = []
            for a in v.arguments:
                vset.append(a)
            vargset.append(vset)

        # first of all, make function application
        origsem = self.copy()
        orige = e.copy()
        pairs = []
        (belowvars, abovevars, bothvars) = self.partition_vars(e)
        ec = e.copy(no_var=True)

        if isinstance(self,LambdaExp) and len(vars)>0:
            compdone = False
        else:
            compdone = True
        varindex = len(vars)-1
        compp = self
        compvars = []
        num_by_comp = 0
        while not compdone:
            current_v = vars[varindex]
            if current_v == self.var and current_v not in abovevars and not current_v.is_event:
                compvars.append(vars[varindex])
                num_by_comp += 1
                p = compp.composition_split(vars, compvars, ec, e)
                ptuple = (p[0], p[1], num_new_lam, num_by_comp)
                pairs.append(ptuple)
                if isinstance(compp.funct,LambdaExp) and\
                        len(vars)>varindex+1:
                    varindex-=1
                    compp = compp.funct
                else: compdone = True
            else: compdone = True

        # all sorts of composition shit in here
        ec = e.copy(no_var=True)
        newvariable = Variable(ec)
        self.replace2(e, newvariable)
        p = self.copy(no_var=True)

        for v in vars:
            nv = Variable(v)
            nv.arguments = v.arguments
            ec.replace2(v, nv)
            v.arguments = []
            newvariable.add_at_front_arg(v)
            new_lambda_exp = LambdaExp()
            new_lambda_exp.set_funct(ec)
            new_lambda_exp.set_var(nv)
            ec = new_lambda_exp

        newvariable.set_type(ec.type())

        new_lambda_exp = LambdaExp()
        new_lambda_exp.set_funct(p)
        new_lambda_exp.set_var(newvariable)
        pair = (new_lambda_exp.copy(), ec.copy(), num_new_lam, 0)
        pairs.append(pair)

        self.replace2(newvariable, e)

        i=0
        for v in vars:
            v.arguments = vargset[i]
            i+=1

        l1 = pair[0].copy()
        e1 = pair[1].copy()
        sem = l1.apply(e1)

        e.repair_binding(orige)
        self.repair_binding(origsem)
        if not sem.equals(self):
            #if self.to_string() != sem.to_string():
            print("sems dont match : "+sem.to_string(True)+"  "+self.to_string(True))
            if len(self.to_string()) == len(sem.to_string()):
                print(self.to_string())
                print(sem.to_string())
                breakpoint()
        else:
            print(111111111)
        return pairs

    def arity(self):
        return 0

    def has_var_order(self, varorder):
        varnum = 0
        for a in self.arguments:
            if isinstance(a,Variable):
                if a.name!=varorder[varnum]:
                    return False
                varnum+=1
        if varnum!=len(varorder):
            return False
        return True

    def var_order(self, L):
        """Omri added 25/7"""
        varnum = 0
        for a in self.arguments:
            if isinstance(a,Variable):
                L[varnum] = a.name
                varnum+=1

    def get_null_pair(self):
        """this should ALWAYS be by composition; parent, child"""
        child = self.copy()
        parent = LambdaExp()

        var = Variable(self)
        parent.set_var(var)
        parent.set_funct(var)
        # all the child cats will have fixed dir and
        # there are no new lambdas in the arg

        # maybe forget the actual direction just the content
        # fixeddircats will actually have the Variables
        fixeddircats = []
        f = self
        done = not isinstance(f,LambdaExp)
        while not done:
            if not isinstance(f,LambdaExp):
                print("not a lambda expression, is  ", f.to_string(True))
            fixeddircats.append(f.var)
            if not isinstance(f.funct,LambdaExp): done = True
            else: f = f.funct

        return (parent, child, 0, 0, None)

    def split_subexp(self, e):
        if self.arity() > 3: return []
        allpairs = []
        self.recalc_parents(True)

        evars = e.unbound_vars()
        # control the arity of the child
        # this may well be problematic
        #if len(evars)>4: return (None, None)
        (orders, num_new_lam, fixeddircats) = self.get_orders(evars)
        for order in orders:
            splits = self.pullout(e, order, num_new_lam)
            for parent_sem, child_sem, num_new_lam, num_by_comp in splits:
                allpairs.append((parent_sem, child_sem, num_new_lam, num_by_comp, fixeddircats))
                # this should be limited, can only do if none by comp
                # parent_sem = splittuple[0]
                # child_sem = splittuple[1]
                if self.allow_type_raise:
                    if num_by_comp==0:
                        if child_sem.can_type_raise():
                            type_raised_child = child_sem.type_raise(parent_sem)
                            print("Type raised child is : "+type_raised_child.to_string(True))
                            print("Parent _sem is : "+parent_sem.to_string(True))
                            trfc = ["typeraised"]
                            trfc.extend(fixeddircats)
                            allpairs.append((type_raised_child, parent_sem.copy(), num_new_lam, 0, trfc))
            if len(order)!=len(evars): print("unmatching varlist lengths")
        #print(len(evars),len(orders))
        return allpairs

    def can_type_raise(self):
        return True

    def type_raise(self, parent):
        v = Variable(parent)
        v.add_arg(self.copy())
        lambda_exp = LambdaExp()
        # it's an opaque way of setting it up,
        # but child is now an argument to which whatever
        # replaces the Variable will be applied
        lambda_exp.set_var(v)
        lambda_exp.set_funct(v)
        return lambda_exp

    def get_orders(self, undervars):
        # if the order is defined by the lambda terms of this
        # thing then go with that order but otherwise we need to
        # get iterations.
        uv2 = []
        evm = None
        for v in undervars:
            if isinstance(v,EventMarker):
                if evm: print("already got event marker")
                evm = v
            else: uv2.append(v)

        fixedorder = []

        for lvar in self.get_lvars(): # all left descendents, I think, why must these be fixed?
            if lvar in undervars:
                fixedorder.append(lvar)
                del uv2[uv2.index(lvar)]

        orderings = []
        if len(uv2)==0:
            ordering = []
            for v in fixedorder: ordering.append(v)
            if evm: ordering.append(evm)
            ordering.reverse()
            orderings.append(ordering)
        else:
            for perm in permutations(uv2):
                ordering = []
                for v in fixedorder: ordering.append(v)
                ordering.extend(perm)
                if evm:
                    ordering.append(evm)
                ordering.reverse()
                orderings.append(ordering)
        return (orderings, len(uv2) +((evm or 0) and 1), fixedorder)

    def get_lvars(self): return []
    # to do in make pairs:
    # 1. want to be able to pull a Variable out in one
    # place only (or in multiple places simultaneously).
    # 2. want to get all subtrees in there. this will
    # cause a ridiculous blowup in complexity...
    #
    # CONSTRAINTS: is across board constraint ok with
    # prepositions????
    # how to do A-over-A constraint???

    def make_pairs(self):
        if self.null_sem(): return []
        rep_pairs = []
        sub_exps = self.all_extractable_sub_exps()
        # ooh, remember this is hard
        # do split, copy then reapply

        for e in sub_exps:
            # this is how we should add null if we're going to
            allow_null = True
            if e==self:
                if allow_null:
                    nullpair = self.get_null_pair()
                    rep_pairs.append(nullpair)
                continue
            if isinstance(e,Variable) or isinstance(e,EventMarker):
                continue
            rep_pairs.extend(self.split_subexp(e))
            if None in rep_pairs:
                breakpoint()
        return rep_pairs

    def null_sem(self):
        return False

    def abstract_over(self, e, v):
        i=0
        for a in self.arguments:
            if a==e:
                self.set_arg(i, v)
        for a in self.arguments:
            a.abstract_over(e, v)

class EmptyExp(Exp):
    def __init__(self):
        self.name = "?"
        self.num_args = 0
        self.arg_types = None
        self.arguments = []
        self.parents = []
        self.arg_set = False
        #self.event=None
        self.is_verb=False
        self.return_type = None
        self.is_null = False
        self.inout = None
        self.double_quant = False

    def copy(self,no_var=False):
        return EmptyExp()

    def make_shell(self, exp_dict):
        if self in exp_dict:
            e = exp_dict[self]
        else:
            e = EmptyExp
        exp_dict[self] = e
        return e

    def is_empty(self):
        print("this is empty")
        return True

    def all_sub_exps(self):
        return []

    def all_extractable_sub_exps(self):
        return []

    def to_string(self, top=True, extra_format=None):
        if self.name=="?":
            self.name="?"+str(Exp.empty_num)
            Exp.empty_num+=1
        s=self.name
        if top:
            Exp.var_num = 0
            Exp.event_num = 0
            Exp.empty_num = 0
        return s

    def equals(self, other):
        return isinstance(other,EmptyExp)

    def equals_placeholder(self, other):
        return self.equals(other)

class Variable(Exp):
    def __init__(self, e):
        self.linked_var = None
        self.name = None
        self.arguments = []
        self.parents = []
        self.is_verb=False
        self.binder = None
        self.equalother = None
        self.varcopy = None
        self.pos_type=None
        self.inout=None
        self.double_quant = False
        self.noun_mod = False
        self.bind_var = None
        self.var_is_const = None
        if e:
            #if e is a quant type of Predicate
            try:
                self.bind_var = e.bind_var
                if self.bind_var:
                    self.var_is_const = e.var_is_const
            except AttributeError:
                pass
            self.t = e.type()
            self.num_args = e.num_args
            self.arg_types = e.arg_types
            self.return_type = e.get_return_type()

            try:
                self.is_event = e.is_event
            except AttributeError:
                self.is_event = False
        else:
            self.num_args = 0
            self.arg_types = []
            # assume that we only introduce entity
            # vars from the corpus
            self.return_type = SemType.e_type()
            self.t = SemType.e_type()
            self.is_event = False
        self.is_null = False

    def copy(self,no_var=False):
        if no_var:
            return self
        if self.varcopy is None:
            newvar = Variable(self)
            self.set_var_copy(newvar)
        v = self.varcopy
        v.linked_var = self.linked_var
        v.var_is_const = self.var_is_const
        v.arguments = []
        if self.arguments:
            v.arguments = [None for a in self.arguments]
            if not self.bind_var or (self.bind_var and self.var_is_const):
                arg0Bound = False
            else:
                arg0Bound = self.arguments[0].binder == self
            # Variable in place of normal Predicate
            args = [None if a is None else a.copy() for a in self.arguments]
            if self.bind_var and arg0Bound:
                # Variable in place of quant with bound Variable
                if not self.var_is_const:
                    newvar = Variable(None)
                    self.arguments[0].set_var_copy(newvar)
                # Variable in place of quant with Constant
                else:
                    newvar = self.arguments[0].copy()
                args[0] = newvar
            for i, a in enumerate(args):
                v.set_arg(i, a)
        return v

    def set_arg_helper(self, position, argument):
        self.arguments.pop(position)
        self.arguments.insert(position, argument)
        if isinstance(argument, Exp):
            argument.add_parent(self)
            self.arg_set = True

    def set_arg(self, position, argument):
        if not self.bind_var:
            self.set_arg_helper(position, argument)
        else:
            if position == 0:
                if isinstance(argument,Variable) and not argument.is_event:
                    if self.var_is_const is None:
                        argument.binder = self
                        self.var_is_const = False
                        self.return_type = SemType.e_type()
                else:
                    if self.var_is_const is None:
                        self.var_is_const = True
                self.set_arg_helper(position, argument)
            if position >= 1:
                if self.var_is_const:
                    for a in argument.all_args():
                        if a.equals(self.arguments[0]):
                            argument.replace2(a, self.arguments[0])
                self.set_arg_helper(position, argument)

    def set_var_in_out(self):
        self.inout = self.binder.inout
        if self.inout is None:
            self.inout = True
        print("self.binder is ", self.binder.to_string(True))
        print("set inout for ", str(id(self)), " to ", self.inout)

    def type(self):
        return self.t

    def semprior(self):
        p = 0.0
        for a in self.arguments:
            p += a.semprior()
        return p

    def vartopprior(self):
        return -2.0

    def is_empty(self):
        return False

    def all_sub_exps(self):
        subexps = [self]
        if len(self.arguments)>0:
            # subexps.append(self)
            for a in self.arguments:
                subexps.extend(a.all_sub_exps())
        return subexps

    def all_extractable_sub_exps(self):
        subexps = []
        if len(self.arguments)>0:
            for a in self.arguments:
                subexps.extend(a.all_extractable_sub_exps())
        return subexps

    def get_all_vars(self, vars):
        if self not in vars:
            vars.append(self)
        for a in self.arguments:
            a.get_all_vars(vars)

    def vars_above(self, other, vars):
        if self==other: return
        if self not in vars:
            vars.append(self)
        for a in self.arguments:
            a.vars_above(other, vars)

    def add_at_front_arg(self, arg):
        self.arguments.insert(0, arg)

    def to_string(self, top=True, extra_format=None):
        if not self.name:
            self.name="UNBOUND"
        s=self.name
        if self.arguments!=[]: s = s+"("
        for a in self.arguments:
            if a is None:
                print("none arg")
                s=s+"NONE"+str(a)
            else:
                s=s+a.to_string(False, extra_format)
            if self.arguments.index(a)<(len(self.arguments)-1):
                s=s+","
        if self.arguments!=[]: s = s+")"

        if top:
            Exp.var_num = 0
            Exp.event_num = 0
            Exp.empty_num = 0
        return s

    def apply(self, other):
        print("cannot apply Variable ", self)
    # checking equality here is tricky because the
    # order of the lambda expressions is important

    # call this whenever introducing a Variable
    def set_equal_to(self, other):
        self.equalother = other

    def set_var_copy(self, other):
        self.varcopy = other

    def equal_type(self, other):
        if not isinstance(other,Variable): return False
        if not other.type().equals(self.type()): return False
        return True

    def set_type(self, t):
        self.t = t

    def equals(self, other):
        if not isinstance(other,Variable): return False
        if len(self.arguments)!=len(other.arguments): return False
        if self.is_event != other.is_event: return False
        # if self and other both bind a Variable, and bound Variables are the first arguments
        # of both self and other, set those Variables to be equal
        binds_var = self.bind_var and not self.var_is_const and len(self.arguments) > 0
        other_binds_var = other.bind_var and not other.var_is_const and len(other.arguments) > 0
        if binds_var and other_binds_var:
            is_binder = self.arguments[0].binder == self
            other_is_binder = other.arguments[0].binder == other
            if is_binder and other_is_binder:
                self.arguments[0].set_equal_to(other.arguments[0])
                other.arguments[0].set_equal_to(self.arguments[0])
        i = 0
        for a in self.arguments:
            if not a.equals(other.arguments[i]): return False
            i+=1
        return other==self.equalother

    def equals_placeholder(self,other):
        if len(self.arguments)!=len(other.arguments): return False
        i = 0
        for a in self.arguments:
            if not a.equals_placeholder(other.arguments[i]): return False
            i+=1
        return other==self.equalother

class LambdaExp(Exp):
    def __init__(self):
        self.linked_var = None
        self.arguments = []
        self.num_args=0
        self.arg_types=[]
        self.parents = []
        self.is_verb=False
        self.return_type = None
        self.is_null = False
        self.pos_type = None
        self.inout=None
        self.double_quant = False
        self.name = "lam"
        pass

    def copy(self,no_var=False):
        lambda_exp = LambdaExp()
        v = Variable(self.var)
        self.var.set_var_copy(v)
        lambda_exp.set_var(v)
        lambda_exp.linked_var = self.linked_var
        f = self.funct.copy(no_var)
        lambda_exp.set_funct(f)
        if self.get_is_null(): lambda_exp.set_is_null()
        return lambda_exp

    # really need to go down from top filling in
    def semprior(self):
        if isinstance(self.funct,Variable):
            return self.funct.vartopprior()
        else:
            return self.funct.semprior()

    def repair_binding(self, orig):
        if orig.var.binder == orig:
            self.var.binder = self
        self.funct.repair_binding(orig.funct)

    def is_q(self):
        return self.funct.is_q()


    def get_lvars(self):
        lvars = [self.var]
        lvars.extend(self.funct.get_lvars())
        return lvars

    def is_conj_v(self):
        return self.funct.is_conj_v()

    def check_if_verb(self):
        return self.funct.check_if_verb()

    def composition_split(self, vars, compvars, ec, e):
        vargset = []
        for v in vars:
            vset = []
            for a in v.arguments: vset.append(a)
            vargset.append(vset)

        vset = []
        self.get_all_vars(vset)

        newvariable = Variable(ec)
        self.replace2(e, newvariable)
        p = self.copy(no_var=True)
        self.replace2(newvariable, e)
        settype=False
        # lambdas are wrong way around
        newvars = []
        for v in vars:
            nv = Variable(v)
            newvars.append(nv)
            nv.arguments = v.arguments
            ec.replace2(v, nv)
            # it is not obvious that this is right
            v.arguments = []

            if v not in compvars: newvariable.add_at_front_arg(v)
            elif not settype:
                newvariable.set_type(ec.type())
                settype=True

            lambda_exp = LambdaExp()
            lambda_exp.set_funct(ec)
            lambda_exp.set_var(nv)
            ec = lambda_exp

        gotp = False
        while not gotp:
            if p.var in compvars:
                p = p.funct
            else: gotp = True
            if not isinstance(p,LambdaExp): gotp = True
        lambda_exp = LambdaExp()
        lambda_exp.set_funct(p)
        lambda_exp.set_var(newvariable)
        pair = (lambda_exp.copy(), ec.copy())

        lambda_exp = lambda_exp.copy()
        ec = ec.copy()

        i = 0
        for v in vars:
            v.arguments = vargset[i]
            i+=1

        vset = []
        return pair

    def all_sub_exps(self):
        subexps = []
        subexps.append(self)
        subexps.extend(self.funct.all_sub_exps())
        if self.funct in subexps: subexps.remove(self.funct)
        return subexps

    def all_extractable_sub_exps(self):
        subexps = []
        subexps.append(self)
        subexps.extend(self.funct.all_extractable_sub_exps())
        if self.funct in subexps:
            subexps.remove(self.funct)
        return subexps

    def get_all_vars(self, vars):
        self.funct.get_all_vars(vars)

    def getheadlambdas(self):
        headlambdas = [self]
        headlambdas.extend(self.funct.getheadlambdas())
        return headlambdas

    def vars_above(self, other, vars):
        if self==other: return
        self.funct.vars_above(other, vars)

    def null_sem(self):
        return self.funct==self.var and len(self.funct.arguments)==0

    def type(self):
        arg_type = self.var.type()
        funct_type = self.funct.type()
        t = SemType(arg_type, funct_type)
        return t

    def set_funct(self, e):
        self.funct = e
        self.return_type = e.get_return_type()
        e.add_parent(self)
        self.arg_set = True

    def set_var(self, var):
        self.var = var
        var.binder = self

    def get_var(self):
        return self.var

    def get_funct(self):
        return self.funct

    def get_deep_funct(self):
        if isinstance(self.funct,LambdaExp): return self.funct
        else: return self.funct.get_deep_funct()

    def arity(self):
        return 1+self.funct.arity()

    def apply(self, e):
        new_exp = None
        var_type = self.var.type()
        arg_type = e.type()
        if var_type.equals(arg_type):
            for a in self.var.arguments:
                if isinstance(e,Variable):
                    e.add_arg(a)
                else:
                    e = e.apply(a)
            if e:
                new_exp = self.funct.replace2(self.var, e)
            return new_exp

    def compose(self, arg):
        if not isinstance(arg,LambdaExp): return None
        sem = self.apply(arg.funct)
        if not sem:
            sem = self.compose(arg.funct)
        if sem:
            arg.set_funct(sem)
            return arg
        else:
            return None

    def args_filled(self):
        return self.funct.args_filled()

    def get_return_type(self):
        return self.type()

    def print_out(self, top, var_num):
        print(self.to_string(top))

    def has_var_order(self, varorder):
        self.var.name = Exp.var_num
        Exp.var_num+=1
        result = self.funct.has_var_order(varorder)
        Exp.var_num=0
        return result

    def set_arg(self, position, pred):
        self.funct.set_arg(position, pred)

    def to_string(self, top=True, extra_format=None):
        self.var.name = "$"+str(Exp.var_num)
        Exp.var_num+=1
        s="lambda "+self.var.name+"_{"+self.var.type().to_string()+"}."+self.funct.to_string(False,extra_format)
        if top:
            Exp.var_num = 0
            Exp.event_num = 0
            Exp.empty_num = 0
        return s

    def equals(self, other):
        if not isinstance(other,LambdaExp) or \
                not other.var.equal_type(self.var):
            return False
        self.var.set_equal_to(other.var)
        other.var.set_equal_to(self.var)
        return other.funct.equals(self.funct)

    def equals_placeholder(self,other):
        if not isinstance(other,LambdaExp) or \
        not other.var.equal_type(self.var):
            return False
        self.var.set_equal_to(other.var)
        other.var.set_equal_to(self.var)
        return other.funct.equals_placeholder(self.funct)

    def replace2(self, e1, e2):
        if self.var == e1:
            self.var = e2

        if self == e1:
            return e2
        self.funct = self.funct.replace2(e1, e2)
        return self

class Neg(Exp):
    def __init__(self, arg, num_args):
        self.name="not"
        self.num_args=num_args
        self.noun_mod = False
        if num_args == 2:
            self.arguments=[arg, EventMarker()]
        else:
            self.arguments=[arg]
        self.arg_types=arg.type()
        self.linked_var = None
        arg.add_parent(self)
        self.parents=[]
        self.arg_set=True
        self.return_type = arg.return_type
        self.is_null = False
        self.pos_type=None
        self.inout=None
        self.double_quant = False
        self.str_shell_prefix = 'not'
        self.str_ubl_prefix = self.name.replace(':','#')

    def copy(self,no_var=False):
        n = Neg(self.arguments[0].copy(no_var), self.num_args)
        if self.num_args == 2:
            n.set_event(self.arguments[1].copy(no_var))
        n.linked_var = self.linked_var
        return n

    def semprior(self):
        return -1.0 + self.arguments[0].semprior()

    def set_event(self, event):
        self.set_arg(1, event)

    def check_if_verb(self):
        return self.arguments[0].check_if_verb()

    def all_extractable_sub_exps(self):
        sub_exps = []
        sub_exps.append(self)
        sub_exps.extend(self.arguments[0].all_extractable_sub_exps())
        return sub_exps

    def all_sub_exps(self):
        sub_exps = []
        sub_exps.append(self)
        sub_exps.extend(self.arguments[0].all_sub_exps())
        return sub_exps

    def type(self):
        return SemType.t_type()

    def equals(self, other):
        if not isinstance(other,Neg): return False
        return other.arguments[0].equals(self.arguments[0])

class EventMarker(Exp):
    def __init__(self, e=None):
        self.name=None
        self.parents=[]
        self.arguments=[]
        self.is_verb=False
        self.binder = None
        self.arg_types=[]
        self.num_args=0
        self.other_event = None
        self.return_type = SemType.event_type()
        self.is_null = False
        self.inout = None
        self.double_quant = False
        if e:
            self.name=e.name

    def copy(self,no_var):
        return self

    def to_string(self, top=True, extra_format=None):
        if not self.name:
            self.name="UNBOUND"
        return self.name

    def all_sub_exps(self):
        return []

    def get_all_vars(self, vars):
        if self not in vars:
            vars.append(self)

    def vars_above(self, other, vars):
        if self==other: return
        if self not in vars:
            vars.append(self)

    def make_shell(self, exp_dict):
        return self

    def replace2(self, e1, e2):
        if self==e1:
            return e2
        return self

    def equals(self, other):
        if not isinstance(other,EventMarker):
            return False
        # always need to have set other_event first
        if self.other_event is None:
            print("other event is None")
            print("comparing to ", other.binder.to_string(True), " which has event ", other.binder.get_event())
            if not self.binder.equals(other.binder):
                return False
        # need to make sure other_event is set
        if not isinstance(other,EventMarker) or \
                not self.other_event==other:
            print("failing on event")
            print("other is ", other, " other_event is ", self.other_event)
            print("this is ", self)
            return False
        return True

    def type(self):
        return SemType.event_type()

class Constant(Exp):
    def copy(self,no_var=False):
        c = Constant(self.name, self.num_args, self.arg_types, self.pos_type)
        c.make_comp_name_set()
        c.linked_var = self.linked_var
        return c

    def set_return_type(self):
        self.return_type = SemType.e_type()

    def type(self):
        return SemType.e_type()

    def make_comp_name_set(self):
        self.names = [self.name]

    def add_comp_name(self, n):
        self.names.append(n)
        self.names.sort()
        self.name=""
        for n in self.names:
            self.name=self.name+n
            if self.names.index(n)<len(self.names)-1:
                self.name=self.name+"+"

    def semprior(self):
        return -1.0

    def make_shell(self, exp_dict):
        if self in exp_dict:
            c = exp_dict[self]
        else:
            c = Constant("placeholder_c", self.num_args, self.arg_types, self.pos_type)
            c.make_comp_name_set()
            exp_dict[self] = c
        return c

    def equals(self, other):
        if not isinstance(other,Constant):
            return False
        if other.name!=self.name:
            return False
        return True

    def add_arg(self, arg):
        print("error, trying to add arg to const")

    def to_string(self, top=True, extra_format=None):
        if extra_format is None:
            return self.name + self._to_string(top, extra_format)
        elif extra_format == 'ubl':
            return self.str_shell_prefix + ':e'
        elif extra_format == 'shell':
            return "placeholder_c"

class Conjunction(Exp):
    def __init__(self):
        self.linked_var = None
        self.num_args = 2
        self.arguments = [EmptyExp(), EmptyExp()]
        self.arg_types=[]
        self.parents = []
        self.return_type = "t"
        self.pos_type="and"
        self.arg_set=False
        self.name="and"
        self.is_null = False
        self.inout = None

    def copy(self,no_var=False):
        c = Conjunction()
        c.linked_var = self.linked_var
        c.set_type(self.name)
        for i, a in enumerate(self.arguments):
            a2 = a.copy(no_var)
            c.set_arg(i, a2)
        return c

    def set_type(self, name):
        self.name = name

    def type(self):
        t = None
        for a in self.arguments:
            if t and t!=a.get_return_type():
                print("bad type for conj,", self.to_string(True), "t was", t.to_string(), "t now", a.type().to_string())
                return None
            else: t = a.get_return_type()
            return t

    def get_return_type(self):
        return self.type()

    def semprior(self):
        p = -1.0
        for a in self.arguments: p += a.semprior()
        return p

    def replace2(self, e1, e2):
        if e1==self:
            return e2
        newargset = []
        for a in self.arguments:
            newargset.append(a.replace2(e1, e2))
        for i, a in enumerate(newargset):
            self.set_arg(i, a)
        return self

    def set_arg(self, position, argument):
        self.arguments[position]=argument

    def check_if_verb(self):
        for a in self.arguments:
            if a.check_if_verb(): return True
        return False

    def has_arg(self, arg):
        for a in self.arguments:
            if a.equals(arg):
                return True
        print("fail on ", arg.to_string(True))
        return False

    def equals(self, other):
        if not isinstance(other,Conjunction):
            return False
        if len(self.arguments)!=len(other.arguments):
            print("conj fail1 ", len(self.arguments), len(other.arguments), "on", self.to_string(True))
            return False
        for a in self.arguments:
            if not other.has_arg(a):
                print("conj fail on", self.to_string(True), "comparing to", other.to_string(True))
                return False
        return True

    def all_extractable_sub_exps(self):
        subexps = [self]
        for a in self.arguments:
            subexps.append(a)
            subexps.extend(a.all_extractable_sub_exps())
        return subexps

    def to_string(self, top=True, extra_format=None):
        if extra_format is None or extra_format == 'shell':
            prefix = 'and('
        elif extra_format == 'ubl':
            prefix = '(and'

        return prefix + self._to_string(top, extra_format)

# Predicates take a number of arguments (not fixed) and
# return a truth value
class Predicate(Exp):
    def __init__(self,name,num_args,arg_types,pos_type,bind_var=False,var_is_const=None,return_type=None):
        self.bind_var = bind_var
        self.var_is_const = var_is_const
        self.onlyinout = None
        self.linked_var = None
        self.name = name
        self.num_args = num_args
        if num_args!=len(arg_types):
            print("error, not right number of args")
        self.arg_types = arg_types
        self.arguments = []
        self.parents = []
        self.str_shell_prefix = 'placeholder_p'
        self.str_ubl_prefix = self.name.replace(':','#')
        for a_t in arg_types:
            self.arguments.append(EmptyExp())
        if return_type:
            self.return_type = return_type
        else:
            if bind_var and not var_is_const:
                self.return_type = SemType.e_type()
            else:
                self.return_type = SemType.t_type()
        self.function_exp = self
        self.pos_type = pos_type
        self.arg_set = False
        self.is_verb=False
        self.is_null = False
        self.inout = None
        self.double_quant = False
        self.str_ubl_prefix = self.name

    def copy(self,no_var=False):
        if not no_var:
            if not self.bind_var:
                args = []
                for a in self.arguments:
                    args.append(None if a is None else a.copy())
                args = [None if a is None else a.copy(no_var) for a in self.arguments]
                e = Predicate(self.name, self.num_args, self.arg_types, self.pos_type, return_type=self.return_type)
            else:
                if not self.var_is_const:
                    newvar = Variable(None)
                    self.arguments[0].set_var_copy(newvar)
                    e = Predicate(self.name, self.num_args, self.arg_types, self.pos_type, bind_var=True, return_type=self.return_type)
                else:
                    newvar = self.arguments[0].copy()
                    e = Predicate(self.name, self.num_args, self.arg_types, self.pos_type, bind_var=True, var_is_const=self.var_is_const, return_type=self.return_type)
                args = [newvar]
                args.extend([a.copy() for a in self.arguments[1:]])

        else:
            if not self.bind_var:
                args = []
                for a in self.arguments:
                    args.append(None if a is None else a.copy(no_var=True))
                args = []
                args = [None if a is None else a.copy(no_var) for a in self.arguments]
                e = Predicate(self.name, self.num_args, self.arg_types, self.pos_type, return_type=self.return_type)
            else:
                if self.var_is_const:
                    args = [a.copy_no_var() for a in self.arguments]
                    e = Predicate(self.name, self.num_args, self.arg_types, self.pos_type, bind_var=True, var_is_const=self.var_is_const, return_type=self.return_type)
                else:
                    args = [self.arguments[0]]
                    args.extend([a.copy_no_var() for a in self.arguments[1:]])
                    e = Predicate(self.name, self.num_args, self.arg_types, self.pos_type, bind_var=True, return_type=self.return_type)
        for i, a in enumerate(args):
            e.set_arg(i, a)
        e.linked_var = self.linked_var
        return e

    def copy(self,no_var=False):
        args = [None if a is None else a.copy(no_var) for a in self.arguments]
        var_is_const_for_copy = True if self.bind_var and self.var_is_const else None
        copied_version = Predicate(self.name, self.num_args, self.arg_types, self.pos_type,
                        return_type=self.return_type, bind_var=self.bind_var,
                        var_is_const=var_is_const_for_copy)
        copied_version.linked_var = self.linked_var
        if self.bind_var:
            if not no_var:
                if self.var_is_const:
                    newvar = self.arguments[0].copy(no_var)
                else:
                    newvar = Variable(None)
                    self.arguments[0].set_var_copy(newvar)
            else:
                newvar = args[0] if self.var_is_const else self.arguments[0]
            args[0] = newvar

        for i, a in enumerate(args):
            copied_version.set_arg(i, a)
        return copied_version

    def set_arg_helper(self, position, argument):
        self.arguments.pop(position)
        self.arguments.insert(position, argument)
        if isinstance(argument, Exp):
            argument.add_parent(self)
            self.arg_set = True

    def set_arg(self, position, argument):
        if not self.bind_var:
            self.set_arg_helper(position, argument)
        else:
            if position == 0:
                if isinstance(argument,Variable):
                    if self.var_is_const is None:
                        argument.binder = self
                        self.var_is_const = False
                        self.return_type = SemType.e_type()
                else:
                    if self.var_is_const is None:
                        self.var_is_const = True
                self.set_arg_helper(position, argument)
            if position >= 1:
                if self.var_is_const:
                    for a in argument.all_args():
                        if a.equals(self.arguments[0]):
                            argument.replace2(a, self.arguments[0])
                self.set_arg_helper(position, argument)

    def all_extractable_sub_exps(self):
        sub_exps = []
        sub_exps.append(self)
        for d in self.arguments:
            arg_sub_exps = d.all_extractable_sub_exps()
            if self.var_is_const:
                if self.arguments[0] in arg_sub_exps and d!=self.arguments[0]:
                    arg_sub_exps = [x for x in arg_sub_exps if x != d]
            for a in arg_sub_exps:
                if a not in sub_exps:
                    sub_exps.append(a)
        return sub_exps

    def semprior(self):
        p = -1.0
        for a in self.arguments: p += a.semprior()
        return p

    def repair_binding(self, orig):
        if self.bind_var and not self.var_is_const:
            if orig.arguments[0].binder == orig:
                self.arguments[0].binder = self
        for arg, orig_arg in zip(self.arguments, orig.arguments):
            arg.repair_binding(orig_arg)

    def get_event(self):
        last_arg = self.arguments[-1]
        if not last_arg: return None
        if not (isinstance(last_arg,EventMarker) or (isinstance(last_arg,Variable) and last_arg.is_event)): return None
        return self.arguments[-1]

    def type(self):
        return self.return_type

    def equals(self, other):
        if not isinstance(other,Predicate) or \
                other.name!=self.name or \
                len(other.arguments)!=len(self.arguments):
            return False
        binds_var = self.bind_var and not self.var_is_const
        other_binds_var = other.bind_var and not other.var_is_const
        if binds_var and other_binds_var:
            self.arguments[0].set_equal_to(other.arguments[0])
            other.arguments[0].set_equal_to(self.arguments[0])
        for i in range(len(self.arguments)):
            if not self.arguments[i].equals(other.arguments[i]):
                return False
        return True

class QMarker(Exp):
    def __init__(self, rep):
        self.linked_var = None
        self.num_args=1
        self.arguments=[rep]
        rep.add_parent(self)
        self.arg_types=[]
        self.parents = []
        self.return_type = "qyn"
        self.pos_type="question"
        self.arg_set=False
        self.name="qyn"
        self.event = None
        self.is_verb = False
        self.is_null = False
        self.inout = None
        self.double_quant = False
        self.noun_mod = False

    def copy(self,no_var=False):
        q = QMarker(self.arguments[0].copy(no_var))
        q.linked_var = self.linked_var
        return q

    def set_event(self, event):
        self.set_arg(1, event)

    def is_q(self):
        return True

    def to_string(self, top=True, extra_format=None):
        #s = "Q("+self.arguments[0].to_string(False,extra_format='ubl')+")"
        s = "Q("+self.arguments[0].to_string(False,extra_format=None)+")"
        if top:
            Exp.var_num = 0
            Exp.event_num = 0
            Exp.empty_num = 0
        return s

    def type(self):
        return SemType.t_type()

    def semprior(self):
        p = -1.0
        for a in self.arguments: p += a.semprior()
        return p

    def equals(self, other):
        if not isinstance(other,QMarker) or \
        not other.arguments[0].equals(self.arguments[0]):
            return False
        return True


def allcombinations(arguments, index, allcombs):
    if index == len(arguments): return
    a = arguments[index]
    newcombs = []
    for comb in allcombs:
        l2 = list(comb)
        l2.append(a)
        newcombs.append(l2)
    allcombs.extend(newcombs)
    allcombs.append([a])
    allcombinations(arguments, index+1, allcombs)

def make_exp(pred_string, exp_string, exp_dict):
    if pred_string in exp_dict:
        e, covered_string = exp_dict[pred_string]
        exp_string_remaining = exp_string if not covered_string else exp_string.split(covered_string)[1]
        return e, exp_string_remaining

    name = pred_string.strip().rstrip()
    name_no_index = re.compile(r"_\d+").split(name)[0]
    pos = name.split("|")[0]
    args, exp_string_remaining = extract_arguments(exp_string, exp_dict)
    print(exp_string, '>>>>>', args)
    arg_types = [x.type() for x in args]
    num_args = len(args)

    if num_args == 0:
        e = Constant(name_no_index, num_args, arg_types, pos)
        e.make_comp_name_set()
    elif num_args in [2, 3]:
        is_quant = check_whether_is_quant(args)
        if pos in ['conj', 'coord']:
            e = Conjunction()
            e.set_type(name)
        elif is_quant:
            e = Predicate(name_no_index, num_args, arg_types, pos, bind_var=True)
        else:
            e = Predicate(name_no_index, num_args, arg_types, pos)
    else:
        e = Predicate(name_no_index, num_args, arg_types, pos)

    for i, arg in enumerate(args):
        e.set_arg(i, arg)

    exp_dict[pred_string] = [e, get_covered_string(exp_string, exp_string_remaining)]
    return e, exp_string_remaining

def get_covered_string(exp_string, exp_string_remaining):
    if exp_string and not exp_string_remaining:
        covered_string = exp_string
    elif exp_string:
        covered_string = exp_string.split(exp_string_remaining)[0]
    else:
        covered_string = ""
    return covered_string

def check_whether_is_quant(args):
    quant_with_var = isinstance(args[0],Variable) and args[0] in args[1].all_sub_exps()
    quant_with_const = isinstance(args[0],Constant) and any([args[0].equals(x) for x in args[1].all_sub_exps()])
    if len(args) == 2:
        return quant_with_var or quant_with_const
    else:
        return False

def make_exp_with_args(exp_string, exp_dict):
    is_lambda = exp_string[:6]=="lambda"
    arguments_present = -1<exp_string.find("(")<exp_string.find(")")
    no_commas = exp_string.find(",")==-1
    commas_inside_parens = -1<exp_string.find("(")<exp_string.find(",")

    if is_lambda:
        e, exp_string_remaining = make_lambda(exp_string, exp_dict)
    elif arguments_present and (commas_inside_parens or no_commas):
        e, exp_string_remaining = make_complex_expression(exp_string, exp_dict)
    else:
        e, exp_string_remaining = make_var_or_const(exp_string, exp_dict)

    return e, exp_string_remaining

def make_lambda(exp_string, exp_dict):
    vname = exp_string[7:exp_string.find("_{")]
    tstring = exp_string[exp_string.find("_{")+2:exp_string.find("}")]
    v = Variable(None)
    t = SemType.make_type(tstring)
    v.t = t
    if tstring == "r":
        v.is_event = True
    exp_dict[vname] = v
    v.name = vname
    exp_string = exp_string[exp_string.find("}.")+2:]
    (f, exp_string_remaining) = make_exp_with_args(exp_string, exp_dict)
    e = LambdaExp()
    e.set_funct(f)
    e.set_var(v)
    return e, exp_string_remaining

def make_complex_expression(exp_string, exp_dict):
    predstring, exp_string = exp_string.split("(", 1)
    if predstring in ["and", "and_comp", "not", "Q"]:
        e, exp_string_remaining = make_log_exp(predstring, exp_string, exp_dict)
    elif predstring[0]=="$":
        e, exp_string_remaining = make_vars(predstring, exp_string, exp_dict)
    else:
        e, exp_string_remaining = make_exp(predstring, exp_string, exp_dict)
    if e is None:
        print("none e for |" + predstring + "|")
    return e, exp_string_remaining

def make_var_or_const(exp_string, exp_dict):
    if ',' in exp_string and ')' in exp_string:
        constend = min(exp_string.find(","), exp_string.find(")"))
    else:
        constend = max(exp_string.find(","), exp_string.find(")"))
    if constend == -1:
        constend = len(exp_string)
    conststring = exp_string[:constend]
    if conststring[0]=="$":
        e, exp_string_remaining = make_vars(conststring, exp_string[constend:], exp_dict, parse_args=False)
    else:
        e, exp_string_remaining = make_exp(conststring, "", exp_dict)
    return e, exp_string_remaining

def extract_arguments(exp_string, exp_dict):
    finished = False if exp_string else True
    num_brack = 1
    i = 0
    j = 0
    arglist = []
    #if exp_string == '$0,Q(not(placeholder_p($1,(and(placeholder_p($1),placeholder_p($1))),$0)))':
        #breakpoint()
    while not finished:
        if num_brack==0:
            finished = True
        elif exp_string[i] in [",", ")"] and num_brack==1:
            if i>j:
                a, _ = make_exp_with_args(exp_string[j:i], exp_dict)
                if not a:
                    print("cannot make Exp for "+exp_string[j:i])
                arglist.append(a)
            j = i+1
            if exp_string[i]==")": finished = True

        elif exp_string[i]=="(": num_brack+=1
        elif exp_string[i]==")": num_brack-=1
        i += 1
    return arglist, exp_string[i:]

def make_vars(predstring,exp_string,vardict,parse_args=True):
    if predstring not in vardict:
        if "_{" in predstring:
            vname = predstring[:predstring.find("_{")]
            tstring = predstring[predstring.find("_{")+2:predstring.find("}")]
        else:
            # Variable bound by a quantifier
            vname = predstring
            tstring = 'e'
        t = SemType.make_type(tstring)
        e = Variable(None)
        e.t = t
        e.name = vname
        vardict[vname] = e
    else:
        e = vardict[predstring]

    if e.num_args == 0 and parse_args:
        args, exp_string = extract_arguments(exp_string, vardict)
        for arg in args:
            e.add_arg(arg)
    return e, exp_string

def make_log_exp(predstring, exp_string, vardict):
    e = None
    if predstring=="and" or predstring=="and_comp":
        e = Conjunction()
        args, exp_string = extract_arguments(exp_string, vardict)
        for i, arg in enumerate(args):
            e.set_arg(i, arg)

    elif predstring=="not":
        Negargs = []
        while exp_string[0]!=")":
            if exp_string[0]==",":
                exp_string = exp_string[1:]
            a, exp_string = make_exp_with_args(exp_string, vardict)
            Negargs.append(a)
        else:
            e = Neg(Negargs[0], len(Negargs))
            if len(Negargs) > 1:
                e.set_event(Negargs[1])
        exp_string = exp_string[1:]

    elif predstring == "Q":
        qargs = []
        while exp_string[0]!=")":
            if exp_string[0]==",":
                exp_string = exp_string[1:]
            a, exp_string = make_exp_with_args(exp_string, vardict)
            qargs.append(a)
        if len(qargs)!=1:
            print(str(len(qargs))+"args for Q")
        else:
            e = QMarker(qargs[0])
        exp_string = exp_string[1:]

    return e, exp_string

def init_clone(class_obj):
    init_params = {k:getattr(class_obj,k) for k in signature(class_obj.__class__.__init__).parameters.keys() if k != 'self' and k in dir(class_obj)}
    copied_version = class_obj.__class__(**init_params)
    return copied_version

def unpack_lists(nested_lists):
    return [nested_lists] + [x for item in nested_lists.arguments for x in unpack_lists(item)]
