totlen = 0
numline = 0
for line in open("rejectedSents.txt"):
    totlen+=len(line.split())
    numline+=1

print float(totlen)/numline
