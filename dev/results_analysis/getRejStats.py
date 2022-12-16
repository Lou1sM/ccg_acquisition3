oneWdict = {}
oneW = 0
for line in open("rejectedSents.txt"):
    if line not in ["","\n"] and line[:5]=="Sent:":
        line = line[5:]
        
        words = line.split()[:-1]
        if len(words)==1:
            oneW += 1
            if not oneWdict.has_key(words[0]):
                oneWdict[words[0]]=0
            oneWdict[words[0]]+=1
#            print words
sortedw = []
for w in oneWdict:
    sortedw.append((oneWdict[w],w))
sortedw.sort()
sortedw.reverse()
for w in sortedw:
    print w[1]," ",w[0]
print oneW
