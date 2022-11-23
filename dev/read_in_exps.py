# read in all expressions from various files
# save all expressions in a dictionary so that they
# can be accessed by name
# read through all of the Eve sentences and build a set
# of templates

# adj - <e,t>
# det - <<e,t>,e>

import exp

def addFromFile(langFile, inFile, templates):
    for line in inFile:
        if line.find("|")!=-1:
            e = exp.makeExp(line)
            if e:
                templates[e.getName()] = e
                if e.getName()[:3]=="aux":
                    if langFile: print("("+e.getName()+"2:t t ev  t)", file=langFile)
                elif e.__class__ in [exp.constant, exp.conjunction]:
                    pass
                elif e.getName()[:4] in ["prep"]:
                    if langFile: print("("+e.getName()+"2:t e ev t)", file=langFile)
                elif e.getName()[:3] == "adv":
                    if langFile: print("("+e.getName()+"1:t ev t)", file=langFile)

                else:
                    if langFile: print("("+e.getName()+"1:t e t)", file=langFile)
    # loc
    e = exp.predicate("eqLoc", 2, ["e", "e"], "eqloc")
    templates["eqLoc"] = e
    e = exp.predicate("evLoc", 2, ["e", "ev"], "evloc")
    templates["evLoc"] = e

def addTransVerbs(langFile, verbFile, templates):
    for line in verbFile:
        if line.find("|")!=-1:
            name = line.strip().rstrip()

            type = name.split("|")[0]
            e = exp.predicate(name, 3, ["e", "e", "ev"], type)
            #e.hasEvent()
            e.setIsVerb()
            templates[e.getName()] = e
            print("("+e.getName()+"3:t e e ev t)", file=langFile)

def addIntransVerbs(langFile, verbFile, templates):
    for line in verbFile:
        if line.find("|")!=-1:
            name = line.strip().rstrip()
            type = name.split("|")[0]
            e = exp.predicate(name, 2, ["e", "ev"], type)
            #e.hasEvent()
            e.setIsVerb()
            templates[e.getName()] = e
            print("("+e.getName()+"2:t e ev t)", file=langFile)

def addDitransVerbs(langFile, verbFile, templates):
    for line in verbFile:
        if line.find("|")!=-1:
            name = line.strip().rstrip()
            type = name.split("|")[0]
            e = exp.predicate(name, 4, ["e", "e", "e", "ev"], type)
            #e.hasEvent()
            e.setIsVerb()
            templates[e.getName()] = e
            print("("+e.getName()+"4:t e e e ev t)", file=langFile)
