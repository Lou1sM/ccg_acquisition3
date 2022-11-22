from errorFunct import error


class MontagueType():
    def __init__(self,string,is_e,is_t,is_atomic):
        self.string_rep = string
        if string in ['e','t','event']:
            self.is_atomic = True
            self.arg_type = self.funct_type = None
        else:
            self.is_atomic = False
            comma_sep_point = get_comma_seperator_point(string)
            self.funct_type = MontagueType(string[:comma_sep_point])
            self.arg_type = MontagueType(string[comma_sep_point:])

    def equals(self,sem):
        if self.is_atomic != sem.is_atomic:
            return False
        if self.is_atomic:
            return self.string_rep == sem.string_rep
        return (self.arg_type.equals(sem.arg_type) and
                self.funct_type.equals(sem.funct_type))

def get_comma_seperator_point(typestring):
    assert typestring[0] == '<'
    typestring = typestring[1:-1]
    leftbrack = 0
    i = 0
    for c in typestring:
        if c=="<": leftbrack+=1
        elif c==">": leftbrack-=1
        elif c=="," and leftbrack==0:
            return i
        i+=1

class eType:
    def __init__(self):
        pass
    def toString(self):
        return "e"
    def toStringUBL(self):
        return "e"
    def isE(self):
        return True
    def isT(self):
        return False
    def isEvent(self):
        return False
    def equals(self, e):
        if e.isE(): return True
        return False
    def atomic(self):
        return True

class tType:
    def __init__(self):
        pass
    def toString(self):
        return "t"
    def toStringUBL(self):
        return "t"
    def isE(self):
        return False
    def isT(self):
        return True
    def isEvent(self):
        return False
    def equals(self, e):
        if e.isT(): return True
        return False
    def atomic(self):
        return True

class eventType:
    def __init__(self):
        pass
    def toString(self):
        return "r"
    def toStringUBL(self):
        return "r"
    def isE(self):
        return False
    def isT(self):
        return False
    def isEvent(self):
        return True
    def equals(self, e):
        if e.isEvent(): return True
        return False
    def atomic(self):
        return True

class SemType:
    e = eType()
    t = tType()
    event = eventType()
    def __init__(self, argType, functType):
        self.argType = argType
        self.functType = functType

    @staticmethod
    def makeType(typestring):
        if typestring=="e": return SemType.e
        elif typestring=="t": return SemType.t
        elif typestring=="r": return SemType.event
        elif typestring[0]!="<":
            print("type error ", typestring)
            error('tye error')
        typestring = typestring[1:-1]
        leftbrack = 0
        i = 0
        for c in typestring:
            if c=="<": leftbrack+=1
            elif c==">": leftbrack-=1
            elif c=="," and leftbrack==0: break
            i+=1
        argstring = typestring[:i]
        functstring = typestring[i+1:]
        t = SemType(SemType.makeType(argstring), SemType.makeType(functstring))
        return t
    def getArity(self):
        if self.atomic(): return 1
        return self.argType.getArity()+self.functType.getArity()
    def isE(self):
        return False
    def isT(self):
        return False
    @staticmethod
    def eType():
        return SemType.e
    @staticmethod
    def tType():
        return SemType.t
    def isEvent(self):
        return False
    @staticmethod
    def eventType():
        return SemType.event
    def getFunct(self):
        return self.functType
    def getArg(self):
        return self.argType
    def equals(self, e):
        if e.isE() or e.isT() or e.isEvent(): return False
        return self.argType.equals(e.argType) and self.functType.equals(e.functType)
    def toString(self):
        return "<"+self.argType.toString()+","+self.functType.toString()+">"
    def toStringUBL(self):
        return "<"+self.argType.toStringUBL()+","+self.functType.toStringUBL()+">"

    def atomic(self):
        return False
