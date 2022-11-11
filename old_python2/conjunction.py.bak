from exp import *

class conjunction(exp):
    def __init__(self):
        self.linkedVar = None
        self.numArgs = 2
        self.arguments = [emptyExp(), emptyExp()]
        self.argTypes=[]
        self.parents = []
        self.returnType = "t"
        self.posType="and"
        self.argSet=False
        self.name="and"
        self.isNull = False
        self.inout = None

    def setType(self,name):
        self.name = name

    def type(self):
        t = None
        for a in self.arguments:
            if t and t!=a.getReturnType():
                print "bad type for conj, ",self.toString(True)," t was ",t.toString()," t now ",a.type().toString()
                return None
            else: t = a.getReturnType()
            return t

    def getReturnType(self):
        return self.type()

    def semprior(self):
        p = -1.0
        for a in self.arguments: p += a.semprior()
        return p

    def makeShell(self, expDict):
        if self in expDict:
            c = expDict[self]
        else:
            c = conjunction()
            c.setType(self.name)
        for i, a in enumerate(self.arguments):
            a2 = a.makeShell(expDict)
            c.setArg(i, a2)
        expDict[self] = c
        return c

    def copy(self):
        c = conjunction()
        c.linkedVar = self.linkedVar
        c.setType(self.name)
        for i, a in enumerate(self.arguments):
            a2 = a.copy()
            c.setArg(i, a2)
        return c

    def copyNoVar(self):
        c = conjunction()
        c.linkedVar = self.linkedVar
        c.setType(self.name)
        for i, a in enumerate(self.arguments):
            a2 = a.copyNoVar()
            c.setArg(i, a2)
        return c

    def addArg(self,arg):
        if isinstance(arg,conjunction):
            for a in arg.arguments:
                self.addArg(a)
                a.remove_parent(arg)
            return
        self.arguments.append(arg)
        arg.add_parent(self)
        self.argSet=True

        return True

    def removeArg(self,arg):
        for i in range(len(self.arguments)):
            a = self.arguments[i]
            if a==arg:
                self.arguments.pop(i)
                return

    def replace2(self,e1,e2):
        if e1==self:
            return e2
        newargset = []
        for a in self.arguments:
            newargset.append(a.replace2(e1,e2))
        for i, a in enumerate(newargset):
            self.setArg(i, a)
        return self

    def setArg(self,position,argument):
        self.arguments[position]=argument

    def checkIfVerb(self):
        for a in self.arguments:
            if a.checkIfVerb(): return True
        return False

    def hasArg(self,arg):
        for a in self.arguments:
            if a.equals(arg):
                return True
        print "fail on ",arg.toString(True)
        return False

    def hasArgP(self,arg):
        for a in self.arguments:
            if a.equalsPlaceholder(arg):
                return True
        print "failP on ",arg.toString(True),"  ",self.toString(True)
        return False

    def equalsPlaceholder(self,other):
        if other.__class__!=conjunction:
            return False
        if len(self.arguments)!=len(other.arguments):
            print "conj fail1 ",len(self.arguments),len(other.arguments)," on ",self.toString(True)
            return False
        for a in self.arguments:
            if not other.hasArgP(a):
                print "conj fail on ",self.toString(True)
                print "comparing to ",other.toString(True)
                return False
        return True

    def equals(self,other):
        if other.__class__!=conjunction:
            return False
        if len(self.arguments)!=len(other.arguments):
            print "conj fail1 ",len(self.arguments),len(other.arguments)," on ",self.toString(True)
            return False
        for a in self.arguments:
            if not other.hasArg(a):
                print "conj fail on ",self.toString(True)
                print "comparing to ",other.toString(True)
                return False
        return True

    def allExtractableSubExps(self):
        subexps = [self]
        for a in self.arguments:
            subexps.append(a)
            subexps.extend(a.allExtractableSubExps())
        return subexps

    def toString(self,top):
        s="and("
        for i in range(len(self.arguments)):
            s=s+self.arguments[i].toString(False)
            if i<len(self.arguments)-1: s=s+","
        s=s+")"
        if top:
            exp.varNum = 0
            exp.eventNum = 0
            exp.emptyNum = 0
            # self.clearNames()
        return s

    def toStringShell(self,top):
        s="and("
        for i in range(len(self.arguments)):
            s=s+self.arguments[i].toStringShell(False)
            if i<len(self.arguments)-1: s=s+","
        s=s+")"
        if top:
            exp.varNum = 0
            exp.eventNum = 0
            exp.emptyNum = 0
            # self.clearNames()
        return s

    def toStringUBL(self,top):
        s="(and "
        for i in range(len(self.arguments)):
            s = s + self.arguments[i].toStringUBL(False)
            if i<len(self.arguments)-1:
                s = s + " "
        s = s + ")"
        if top:
            exp.varNum = 0
            exp.eventNum = 0
            exp.emptyNum = 0
            # self.clearNames()
        return s