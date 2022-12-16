import cPickle as pickle
import sys

repo = pickle.load(open(sys.argv[1]))
for x in repo.verb_instances:
    print(str(x[0])+'\t'+str(x[2]))


    
