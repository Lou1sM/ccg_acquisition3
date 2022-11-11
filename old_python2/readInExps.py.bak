# read in all expressions from various files
# save all expressions in a dictionary so that they
# can be accessed by name
# read through all of the Eve sentences and build a set
# of templates


# adj - <e,t>
# det - <<e,t>,e>

from exp import *
from conjunction import *
from constant import *
# from quant import *
from predicate import *
import expFunctions

def addFromFile(langFile,inFile,templates):
	for line in inFile:
		if line.find("|")!=-1:
			e = expFunctions.makeExp(line)
			if e:
				templates[e.getName()] = e
				if e.getName()[:3]=="aux":
					if langFile: print >> langFile,"("+e.getName()+"2:t t ev  t)"
				elif e.__class__ in [constant,conjunction]:
					pass
				elif e.getName()[:4] in ["prep"]:
					if langFile: print >> langFile,"("+e.getName()+"2:t e ev t)"
				elif e.getName()[:3] == "adv":
					if langFile: print >> langFile,"("+e.getName()+"1:t ev t)"
					
				else:
					if langFile: print >> langFile,"("+e.getName()+"1:t e t)"
	# loc
	e = predicate("eqLoc",2,["e","e"],"eqloc")
	templates["eqLoc"] = e
	e = predicate("evLoc",2,["e","ev"],"evloc")
	templates["evLoc"] = e

def addTransVerbs(langFile,verbFile,templates):
	for line in verbFile:
		if line.find("|")!=-1:
			name = line.strip().rstrip()
			
			type = name.split("|")[0]
			e = predicate(name,3,["e","e","ev"],type)
			#e.hasEvent()
			e.setIsVerb()
			templates[e.getName()] = e
			print >> langFile,"("+e.getName()+"3:t e e ev t)"

def addIntransVerbs(langFile,verbFile,templates):
	for line in verbFile:
		if line.find("|")!=-1:
			name = line.strip().rstrip()
			type = name.split("|")[0]
			e = predicate(name,2,["e","ev"],type)
			#e.hasEvent()
			e.setIsVerb()
			templates[e.getName()] = e
			print >> langFile,"("+e.getName()+"2:t e ev t)"
def addDitransVerbs(langFile,verbFile,templates):
	for line in verbFile:
		if line.find("|")!=-1:
			name = line.strip().rstrip()
			type = name.split("|")[0]
			e = predicate(name,4,["e","e","e","ev"],type)
			#e.hasEvent()
			e.setIsVerb()
			templates[e.getName()] = e
			print >> langFile,"("+e.getName()+"4:t e e e ev t)"
