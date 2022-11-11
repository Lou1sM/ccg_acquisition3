from exp import *
# from eventMarker import *

class neg(exp):
    def __init__(self,arg,numArgs):
        self.name="not"
        self.numArgs=numArgs
        self.nounMod = False
        if numArgs == 2:
            self.arguments=[arg,eventMarker()]
        else:
            self.arguments=[arg]
            # self.nounMod = arg.isNounMod()
        self.argTypes=arg.type()
        self.linkedVar = None
        arg.add_parent(self)
        self.parents=[]
        self.argSet=True
        # self.returnType = semType.tType()
        self.returnType = arg.returnType
        self.isNull = False
        self.posType=None
        self.inout=None
        self.doubleQuant = False
        #self
        #self.event = None

    def semprior(self):
        return -1.0 + self.arguments[0].semprior()

    def makeShell(self, expDict):
        if self in expDict:
            n = expDict[self]
        else:
            n = neg(self.arguments[0].makeShell(expDict), self.numArgs)
            if self.numArgs == 2:
                n.setEvent(self.arguments[1].makeShell(expDict))
        expDict[self] = n
        return n

    def copy(self):
        #print "copying ",self.toString(True)
        n = neg(self.arguments[0].copy(), self.numArgs)
        if self.numArgs == 2:
            n.setEvent(self.arguments[1].copy())
        n.linkedVar = self.linkedVar
        return n

    def copyNoVar(self):
        n = neg(self.arguments[0].copyNoVar(), self.numArgs)
        if self.numArgs == 2:
            n.setEvent(self.arguments[1].copyNoVar())
        n.linkedVar = self.linkedVar
        return n

    def toStringShell(self,top):
        s="not"
        #if self.checkIfVerb():
            ##self.getEvent().setName("e"+str(exp.eventNum))
            #exp.eventNum+=1

        #print "tring for ",self.name
        if len(self.arguments)>0: s=s+"("
        for a in self.arguments:
            if isinstance(a,exp): s=s+str(a.toStringShell(False))
            if self.arguments.index(a)<self.numArgs-1: s=s+","
        if len(self.arguments)>0: s=s+")"
        #if self.event:
            #s=s+":"+self.event.toString(False)
        if top:
            exp.varNum = 0
            exp.eventNum = 0
            exp.emptyNum = 0
        #print "returning "+s
        return s

    def setEvent(self,event):
        self.setArg(1,event)

    def checkIfVerb(self):
        return self.arguments[0].checkIfVerb()

    def allExtractableSubExps(self):
        subExps = []
        subExps.append(self)
        subExps.extend(self.arguments[0].allExtractableSubExps())
        return subExps

    def allSubExps(self):
        subExps = []
        subExps.append(self)
        subExps.extend(self.arguments[0].allSubExps())
        return subExps

    def type(self):
        return semType.tType()

    def equalsPlaceholder(self,other):
        if other.__class__!=neg: return False
        return other.arguments[0].equalsPlaceholder(self.arguments[0])

    def equals(self,other):
        if other.__class__!=neg: return False
        return other.arguments[0].equals(self.arguments[0])