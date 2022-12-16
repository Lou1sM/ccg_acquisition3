

c1 = {}

for line in open('childes-data-oldCats.txt'):
    if line!='':
        if line[0]!='#':
            if line.find('sent: ')!=-1:
                sent = line.rstrip()
            elif line.find('sem: ')!=-1:
                sem = line.rstrip()
                pass
            elif line.find('head:')!=-1:
                head = line.rstrip()
                c1[sent]=(sem,head)
                pass

for line in open('childes-data.txt'):
    if line!='':
        if line[0]!='#':
            if line.find('sent: ')!=-1:
                sent = line.rstrip()
                if c1.has_key(sent):
                    print sent
                    print c1[sent][0]
                    print c1[sent][1]
                    print 'example_end\n'
