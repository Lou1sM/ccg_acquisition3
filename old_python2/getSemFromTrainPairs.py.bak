from exp import exp
import expFunctions

for line in open("trainPairs.txt"):
    if line[:4]=="Sem:":
        semstring = line[4:].strip().rstrip()
        if not semstring.__contains__("^") and not semstring.__contains__("sk "):
            print "\n\nmaking sem for : "+semstring
            r = expFunctions.makeExpWithArgs(semstring,{})
            if r:
                print "r is ",r
                e = r[0]
                print "made : "+e.toString(True)
            else: print "failed on ^"
        
        
    
