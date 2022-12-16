import re

nouns_re = re.compile('n\\|[A-Za-z]+')

def runon(fn,counter,all_nouns,total_nouns):
    f = open(fn)
    for line in f:
        if line.strip().startswith('Cat :'):
            counter += 1
            cur_nouns = nouns_re.findall(line.strip())
            for n in cur_nouns:
                if n not in all_nouns:
                    all_nouns.append(n)
            total_nouns.append((counter,len(all_nouns)))
    return counter, all_nouns, total_nouns

def main(prefix):
    counter = 0
    all_nouns = []
    total_nouns = []
    for i in range(1,20):
        counter,all_nouns,total_nouns = \
            runon(prefix+str(i),counter,all_nouns,total_nouns)
    #for ind,entry in total_nouns:
    #    print(str(ind)+'\t'+str(entry))
    return [(0,0)] + total_nouns

if __name__ == '__main__':
    main("new_outputs/train_parses_1W_")


