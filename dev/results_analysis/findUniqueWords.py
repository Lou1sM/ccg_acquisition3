trainwords = set()
for line in open("trainPairsTrain.txt"):
    if line[:5]=="Sent:":
        for w in line[6:].split(): trainwords.add(w)

numsent = 0
numnotseen = 0
for line in open("testPairs.txt"):
    notseen = False
    if line[:5]=="Sent:":
        for w in line[6:].split():
            if not w in trainwords:
                print w," not seen"
                notseen = True
        if notseen: numnotseen += 1
        numsent += 1
print numnotseen," never seen"
print "out of ",numsent," sentences"
