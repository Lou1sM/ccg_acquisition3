from exp import *
# from eventMarker import *

# predicates take a number of arguments (not fixed) and
# return a truth value
class predicate(exp):
    def __init__(self,name,numArgs,argTypes,posType,bindVar=False,varIsConst=None,args=[], returnType=None):
        self.bindVar = bindVar
        self.varIsConst = varIsConst
        self.onlyinout = None
        self.linkedVar = None
        self.name = name
        self.numArgs = numArgs
        if numArgs!=len(argTypes):
            print "error, not right number of args"
        self.argTypes = argTypes
        self.arguments = []
        self.parents = []

        for aT in argTypes:
            self.arguments.append(emptyExp())
        # for i,a in enumerate(args):
        #     self.setArg(i,a)

        if returnType:
            self.returnType = returnType
        else:
            if bindVar and not varIsConst:
                self.returnType = semType.eType()
            else:
                self.returnType = semType.tType()
            # if args[-1].__class__ == variable and args[-1].isEvent:
            #     self.returnType = semType.tType()
            # else:
            #     self.returnType = semType.eType()

        # self.setReturnType()
        self.functionExp = self
        # self.nounMod = False
        self.posType = posType
        self.argSet = False
        self.isVerb=False
        self.isNull = False
        self.inout = None
        self.doubleQuant = False
        self.string = ""

    def setArgHelper(self, position, argument):
        self.arguments.pop(position)
        self.arguments.insert(position,argument)
        if isinstance(argument,exp):
            argument.add_parent(self)
            self.argSet = True

    def setArg(self,position,argument):
        if not self.bindVar:
            self.setArgHelper(position, argument)
        else:
            if position == 0:
                if argument.__class__ == variable:
                    if self.varIsConst == None:
                        argument.setBinder(self)
                        self.varIsConst = False
                        self.returnType = semType.eType()
                else:
                    if self.varIsConst == None:
                        self.varIsConst = True
                self.setArgHelper(position, argument)
            if position >= 1:
                if self.varIsConst:
                    for a in argument.allArgs():
                        if a.equals(self.arguments[0]):
                            argument.replace2(a, self.arguments[0])
                self.setArgHelper(position, argument)

    def allExtractableSubExps(self):
        subExps = []
        subExps.append(self)
        for d in self.arguments:
            arg_subExps = d.allExtractableSubExps()
            if self.varIsConst:
                if self.arguments[0] in arg_subExps and d!=self.arguments[0]:
                    arg_subExps = [x for x in arg_subExps if x != d]
            for a in arg_subExps:
                if a not in subExps:
                    subExps.append(a)
        return subExps

    # def setReturnType(self):
    #     if self.bindVar and not self.varIsConst:
    #         self.returnType = semType.eType()
    #     else:
    #         self.returnType = semType.tType()

    def semprior(self):
        p = -1.0
        for a in self.arguments: p += a.semprior()
        return p

    def makeShell(self, expDict):
        args = []
        for a in self.arguments:
            args.append(a.makeShell(expDict))
        if self in expDict:
            e = expDict[self]
        elif self.bindVar and len(args) > 1:
            e = predicate("placeholderP",self.numArgs,self.argTypes,self.posType,
                          bindVar=self.bindVar, returnType=self.returnType)
        elif self.bindVar:
            e = predicate("placeholderP",self.numArgs,self.argTypes,self.posType,
                          bindVar=self.bindVar,varIsConst=self.varIsConst, returnType=self.returnType)
        else:
            e = predicate("placeholderP",self.numArgs,self.argTypes,self.posType, returnType=self.returnType)
        i=0
        for a in args:
            e.setArg(i,a)
            i+=1
        expDict[self] = e
        return e

    def copy(self):
        if not self.bindVar:
            args = []
            for a in self.arguments:
                args.append(a.copy())
            e = predicate(self.name,self.numArgs,self.argTypes,self.posType, returnType=self.returnType)
            e.linkedVar = self.linkedVar
            for i, a in enumerate(args):
                e.setArg(i,a)
        else:
            if not self.varIsConst:
                newvar = variable(None)
                self.arguments[0].setVarCopy(newvar)
                e = predicate(self.name,self.numArgs,self.argTypes,self.posType,bindVar=True, returnType=self.returnType)
            else:
                newvar = self.arguments[0].copy()
                e = predicate(self.name,self.numArgs,self.argTypes,self.posType,bindVar=True,varIsConst=self.varIsConst, returnType=self.returnType)
            args = [newvar]
            args.extend([a.copy() for a in self.arguments[1:]])
            for i, a in enumerate(args):
                e.setArg(i,a)
            e.linkedVar = self.linkedVar
        return e

    def copyNoVar(self):
        if not self.bindVar:
            args = []
            for a in self.arguments:
                args.append(a.copyNoVar())
            e = predicate(self.name,self.numArgs,self.argTypes,self.posType,returnType=self.returnType)
            e.linkedVar = self.linkedVar
            i=0
            for a in args:
                e.setArg(i,a)
                i+=1
        else:
            if self.varIsConst:
                args = [a.copyNoVar() for a in self.arguments]
                e = predicate(self.name,self.numArgs,self.argTypes,self.posType,bindVar=True,varIsConst=self.varIsConst, returnType=self.returnType)
            else:
                args = [self.arguments[0]]
                args.extend([a.copyNoVar() for a in self.arguments[1:]])
                e = predicate(self.name,self.numArgs,self.argTypes,self.posType,bindVar=True, returnType=self.returnType)
            for i, a in enumerate(args):
                e.setArg(i,a)
            e.linkedVar = self.linkedVar
        return e

    def repairBinding(self, orig):
        if self.bindVar and not self.varIsConst:
            if orig.arguments[0].binder == orig:
                self.arguments[0].setBinder(self)
        for arg, orig_arg in zip(self.arguments, orig.arguments):
            arg.repairBinding(orig_arg)

    def getEvent(self):
        lastArg = self.arguments[-1]
        if not lastArg: return None
        if not (lastArg.__class__==eventMarker or (lastArg.__class__==variable and lastArg.isEvent)): return None
        return self.arguments[-1]

    # this may need a little thinking
    def type(self):
        return self.returnType
        # return semType.tType()

    def equalsPlaceholder(self,other):
        if other.__class__ != predicate or \
        (other.name!=self.name and not (("placeholderP" in self.name) or ("placeholderP" in other.name))) or \
        len(other.arguments)!=len(self.arguments):
                return False
        for i in range(len(self.arguments)):
            if not self.arguments[i].equalsPlaceholder(other.arguments[i]):
                return False
        return True

    def equals(self,other):
        if other.__class__ != predicate or \
        other.name!=self.name or \
        len(other.arguments)!=len(self.arguments):
                return False
        bindsVar = self.bindVar and not self.varIsConst
        other_bindsVar = other.bindVar and not other.varIsConst
        if bindsVar and other_bindsVar:
            self.arguments[0].setEqualTo(other.arguments[0])
            other.arguments[0].setEqualTo(self.arguments[0])
        for i in range(len(self.arguments)):
            if not self.arguments[i].equals(other.arguments[i]):
                return False
        return True

    def toString(self,top):
        s=self.name
        if len(self.arguments)>0: s=s+"("
        for a in self.arguments:
            if isinstance(a,exp): s=s+str(a.toString(False))
            if self.arguments.index(a)<self.numArgs-1: s=s+","
        if len(self.arguments)>0: s=s+")"
        if top:
            exp.varNum = 0
            exp.eventNum = 0
            exp.emptyNum = 0
        return s

    def toStringShell(self,top):
        s="placeholderP"
        if len(self.arguments)>0: s=s+"("
        for a in self.arguments:
            if isinstance(a,exp): s=s+str(a.toStringShell(False))
            if self.arguments.index(a)<self.numArgs-1: s=s+","
        if len(self.arguments)>0: s=s+")"
        if top:
            exp.varNum = 0
            exp.eventNum = 0
            exp.emptyNum = 0
        return s

    def toStringUBL(self,top):
        s=self.name
        if len(self.arguments)>0: s="("+s+str(len(self.arguments))+":t "
        for a in self.arguments:
            if isinstance(a,exp): s=s+str(a.toStringUBL(False))
            if self.arguments.index(a)<self.numArgs-1: s=s+" "
        if len(self.arguments)>0: s=s+")"
        if top:
            exp.varNum = 0
            exp.eventNum = 0
            exp.emptyNum = 0
        return s