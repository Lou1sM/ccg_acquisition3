input = open('training_pairs.txt','r')
output = open('t','w')
i = 0
for line in input:
    if i == 10:
        abort()
    if line.find('sent:')!=-1:
        i+=1
    print >> output, line,
        
