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

class EType:
    def __init__(self):
        pass
    def to_string(self):
        return "e"
    def to_string_uBL(self):
        return "e"
    def is_e(self):
        return True
    def is_t(self):
        return False
    def is_event(self):
        return False
    def equals(self, e):
        if e.is_e(): return True
        return False
    def atomic(self):
        return True

class TType:
    def __init__(self):
        pass
    def to_string(self):
        return "t"
    def to_string_uBL(self):
        return "t"
    def is_e(self):
        return False
    def is_t(self):
        return True
    def is_event(self):
        return False
    def equals(self, e):
        if e.is_t(): return True
        return False
    def atomic(self):
        return True

class EventType:
    def __init__(self):
        pass
    def to_string(self):
        return "r"
    def to_string_uBL(self):
        return "r"
    def is_e(self):
        return False
    def is_t(self):
        return False
    def is_event(self):
        return True
    def equals(self, e):
        if e.is_event(): return True
        return False
    def atomic(self):
        return True

class SemType:
    e = EType()
    t = TType()
    event = EventType()
    def __init__(self, arg_type, funct_type):
        self.arg_type = arg_type
        self.funct_type = funct_type

    @staticmethod
    def make_type(typestring):
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
        t = SemType(SemType.make_type(argstring), SemType.make_type(functstring))
        return t
    def get_arity(self):
        if self.atomic(): return 1
        return self.arg_type.get_arity()+self.funct_type.get_arity()
    def is_e(self):
        return False
    def is_t(self):
        return False
    @staticmethod
    def e_type():
        return SemType.e
    @staticmethod
    def t_type():
        return SemType.t
    def is_event(self):
        return False
    @staticmethod
    def event_type():
        return SemType.event
    def get_funct(self):
        return self.funct_type
    def get_arg(self):
        return self.arg_type
    def equals(self, e):
        if e.is_e() or e.is_t() or e.is_event(): return False
        return self.arg_type.equals(e.arg_type) and self.funct_type.equals(e.funct_type)
    def to_string(self):
        return "<"+self.arg_type.to_string()+","+self.funct_type.to_string()+">"
    def to_string_uBL(self):
        return "<"+self.arg_type.to_string_uBL()+","+self.funct_type.to_string_uBL()+">"

    def atomic(self):
        return False
