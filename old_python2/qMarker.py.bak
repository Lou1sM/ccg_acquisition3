from exp import *
# from eventMarker import *

class qMarker(exp):
    def __init__(self,rep):
        #print "making Q for ",rep.toString(True)
        # second arg is event
        self.linkedVar = None
        self.numArgs=1
        self.arguments=[rep]
        rep.add_parent(self)
        self.argTypes=[]
        self.parents = []
        self.returnType = "qyn"
        self.posType="question"
        self.argSet=False
        self.name="qyn"
        self.event = None
        self.isVerb = False
        self.isNull = False
        self.inout = None
        self.doubleQuant = False
        self.nounMod = False

    def setEvent(self,event):
        self.setArg(1,event)

    def isQ(self):
        return True

    def toString(self,top):
        # s = "Q("+self.arguments[0].toString(False)+","+self.arguments[1].toString(False)+")"
        s = "Q("+self.arguments[0].toStringUBL(False)+")"
        if top:
            exp.varNum = 0
            exp.eventNum = 0
            exp.emptyNum = 0
            # self.clearNames()
        return s

    def toStringShell(self,top):
        # s = "Q("+self.arguments[0].toStringShell(False)+","+self.arguments[1].toStringShell(False)+")"
        s = "Q("+self.arguments[0].toStringUBL(False)+")"
        if top:
            exp.varNum = 0
            exp.eventNum = 0
            exp.emptyNum = 0
            # self.clearNames()
        return s

    def toStringUBL(self,top):
        # s = "(Q:t "+self.arguments[0].toStringUBL(False)+" "+self.arguments[1].toStringUBL(False)+")"
        s = "(Q:t "+self.arguments[0].toStringUBL(False)+")"
        if top:
            exp.varNum = 0
            exp.eventNum = 0
            exp.emptyNum = 0
            # self.clearNames()
        return s

    def type(self):
        return semType.tType()

    def semprior(self):
        p = -1.0
        for a in self.arguments: p += a.semprior()
        return p

    def makeShell(self, expDict):
        if self in expDict:
            q = expDict[self]
        else:
            q = qMarker(self.arguments[0].makeShell(expDict))
        expDict[self] = q
        # q.setEvent(self.arguments[1].makeShell())
        return q

    def copy(self):
        #print "copying ",self.toString(True)
        q = qMarker(self.arguments[0].copy())
        q.linkedVar = self.linkedVar
        # q.setEvent(self.arguments[1].copy())
        return q

    def copyNoVar(self):
        q = qMarker(self.arguments[0].copyNoVar())
        q.linkedVar = self.linkedVar
        # q.setEvent(self.arguments[1].copyNoVar())
        return q

    def equals(self,other):
        if other.__class__ != qMarker or \
        not other.arguments[0].equals(self.arguments[0]):
            return False
        return True

    def equalsPlaceholder(self,other):
        if other.__class__ != qMarker or \
        not other.arguments[0].equalsPlaceholder(self.arguments[0]):
            return False
        return True