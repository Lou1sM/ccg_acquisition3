# this is a simple script to introduce noise by pairing the current
# sentence with the adjacent ones.
sents = []
sems = []
infile = "data/trainPairs.adam.ready"
numreps = 7

for session_no in range(1, 42):
    for line in open(infile+"_"+str(session_no)):
        line = line.strip().rstrip()
        if line[:5]=="Sent:": sents.append(line)
        if line[:4]=="Sem:": sems.append(line)

    output = open(infile+str(numreps)+"reps_"+str(session_no), "w")
    i=0
    for sent in sents:
        print(sent, file=output)
        for j in range(numreps):
            index=(i-int(numreps/2)+j)%len(sems)
            print(sems[index], file=output)
        i+=1
        print("example_end\n", file=output)
