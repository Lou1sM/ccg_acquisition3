pos_marking_dict = {
                    'adj':set(['N|N']),
                    'adv':set(['S|S','(S|NP)|(S|NP)|(S|NP|NP)|(S|NP|NP)']),
                    'adv:int':set([None]),
                    'adv:tem':set([None]),
                    'aux':set([None]),
                    'chi':set([None]),
                    #'co':set(['S']),
                    'co':set([None]),
                    'conj':set(['S|S|S','NP|NP|NP','N|N|N','(S|NP)|(S|NP)|(S|NP)']), # don't have schemas yet]), minimal list for now
                    'coord':set(['S|S|S','NP|NP|NP','N|N|N','(S|NP)|(S|NP)|(S|NP)']), # same as above ok for now?
                    'cop':set(['S|S','X']),
                    'det':set(['NP|N']),
                    'det:art':set(['NP|N']),
                    'det:dem':set(['NP|N']),
                    'det:int':set(['NP']),
                    'det:num':set(['NP|N']),
                    'det:poss':set(['NP|N']),
                    'meta':set([None]),
                    'mod':set(['S|NP|(S|NP)','S|(S|NP)|NP','S|NP|(S|NP)','S|(S|NP)|NP']),
                    'mod:aux':set([None]),
                    'n':set(['N']),
                    'n:gerund':set(['NP']),
                    'n:let':set(['NP']),
                    'n:prop':set(['NP']),
                    'n:pt':set(['N']), # seems to be for nouns with plural morph.]), like scissors
                    #'neg':set(['S|S','(S|NP)|(S|NP)','(S|NP|NP)|(S|NP|NP)']), # ?
                    'on':set([None]),
                    'part':set([None]),
                    'poss':set([None]),
                    'post':set(['S|S']), # ?
                    'prep':set(['S|NP|NP', 'S|S|NP','NP|NP|NP']),
                    'pro:dem':set(['NP']), # ?
                    'pro:exist':set(['NP']),
                    'pro:indef':set(['NP']),
                    'pro:int':set(['NP']),
                    'pro:obj':set(['NP']),
                    'pro:per':set(['NP']),
                    'pro:poss':set(['NP']),
                    'pro:refl':set(['NP']),
                    'pro:rel':set(['NP']),
                    'pro:sub':set(['NP']),
                    'qn':set(['NP|N']),
                    'sing':set([None]),
                    'v':set(['S|NP','S|NP|NP']),
                    #'v:obj':set([None]),
                    'wplay':set([None]),
                    }
pos_marking_dict['neg'] = set([f'({x})|({x})' for x in pos_marking_dict['v']] +[f'({x})|({x})' for x in pos_marking_dict['mod']] )

pos_marking_dict = {k:set(['X']) if v == set([None]) else v for k,v in pos_marking_dict.items()}


base_lexicon = {k:set(['NP']) for k in ('you','i','me','he','she','it','WHAT','WHO')}
base_lexicon['equals'] = set(['S|NP|NP'])
base_lexicon['hasproperty'] = set(['S|NP|(N|N)'])
base_lexicon['mod|will_2'] = set(['S|NP', 'S|NP|NP'])
base_lexicon['not'] = set(['X'])

gt_word2lfs = {
            "'ll": 'lambda $0.lambda $1.mod|~will ($0 $1)',
             "'re": 'lambda $0.lambda $1.Q (v|equals $0 $1)',
             "'s": 'lambda $0.lambda $1.v|equals $1 $0',
             'Adam': 'n:prop|adam',
             'I': 'pro:sub|i',
             'a': 'lambda $0.det:art|a $0',
             'alright': 'adj|alright',
             'an': 'lambda $0.det:art|a $0',
             'another': 'lambda $0.qn|another $0',
             'are': 'lambda $0.lambda $1.Q (v|equals $0 $1)',
             'break': 'lambda $0.lambda $1.v|break $1 $0',
             'can': 'lambda $0.lambda $1.Q (mod|can ($1 $0))',
             'd': 'lambda $0.lambda $1.Q (mod|do ($1 $0))',
             'did': 'lambda $0.v|do-past $0',
             'do': 'lambda $0.lambda $1.Q (mod|do ($1 $0))',
             'dropped': 'lambda $0.lambda $1.v|drop-past $1 $0',
             'good': 'adj|good',
             'have': 'lambda $0.lambda $1.v|have $1 $0',
             'he': 'pro:sub|he',
             'his': 'lambda $0.det:poss|his $0',
             'is': 'lambda $0.lambda $1.Q (v|equals $0 $1)',
             'it': 'pro:per|it',
             'know': 'lambda $0.lambda $1.v|know $1 $0',
             'like': 'lambda $0.lambda $1.v|like $1 $0',
             'lost': 'lambda $0.lambda $1.v|lose-past $1 $0',
             'may': 'lambda $0.lambda $1.mod|may ($0 $1)',
             'missed': 'lambda $0.v|miss-past $0',
             'my': 'lambda $0.det:poss|my $0',
             'name': 'n|name',
             'need': 'lambda $0.lambda $1.v|need $1 $0',
             'nice': 'lambda $0.adv|very $0',
             'not': 'lambda $0.lambda $1.not (v|equals $1 $0)',
             'one': 'pro:indef|one',
             'pencil': 'n|pencil',
             'right': 'lambda $0.Q (v|hasproperty $0 adj|right)',
             'say': 'lambda $0.lambda $1.Q (v|say $0 $1)',
             'see': 'lambda $0.lambda $1.Q (v|see $0 $1)',
             'some': 'lambda $0.qn|some $0',
             'that': 'pro:dem|that',
             'the': 'lambda $0.det:art|the $0',
             'they': 'pro:sub|they',
             'this': 'pro:dem|this',
             'very': 'lambda $0.v|hasproperty $0 adj|nice',
             'was': 'lambda $0.lambda $1.Q (v|equals $0 $1)',
             'we': 'pro:sub|we',
             'what': 'pro:int|WHAT',
             'who': 'pro:int|WHO',
             'you': 'pro:per|you',
             'your': 'lambda $0.det:poss|your $0',
             'them': 'pro:obj|them',
             }


