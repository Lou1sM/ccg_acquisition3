# types
from errorFunct import error

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
	def equals(self,e):
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
	def equals(self,e):
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
	def equals(self,e):
		if e.isEvent(): return True
		return False
	def atomic(self):
		return True
	#def getArg(self):
		#return None
		
		
class semType:
	e = eType()
	t = tType()
	event = eventType()
	def __init__(self,argType,functType):
		#eType = "e"
		#tType = "t"		
		self.argType = argType
		self.functType = functType
	@staticmethod
	def makeType(typestring):
		if typestring=="e": return semType.e
		elif typestring=="t": return semType.t
		elif typestring=="r": return semType.event
		# elif typestring=="ev": return semType.event
		elif typestring[0]!="<":
			print "type error ",typestring
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
		t = semType(semType.makeType(argstring),semType.makeType(functstring))
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
		return semType.e
	@staticmethod
	def tType():
		return semType.t
	def isEvent(self):
		return False
	@staticmethod
	def eventType():
		return semType.event
	def getFunct(self):
		return self.functType
	def getArg(self):
		return self.argType
	def equals(self,e):
		#print "comparing ",self.toString(),
		#print " to ",e.toString()
		if e.isE() or e.isT() or e.isEvent(): return False
		return self.argType.equals(e.argType) and self.functType.equals(e.functType)
	def toString(self):
		return "<"+self.argType.toString()+","+self.functType.toString()+">"
	def toStringUBL(self):
		return "<"+self.argType.toStringUBL()+","+self.functType.toStringUBL()+">"

	def atomic(self):
		return False
		
	


		
