pos_marking_dict = {
                    'adj':set(['N|N']),
                    'adv':set([None]),
                    'adv:int':set([None]),
                    'adv:tem':set([None]),
                    'aux':set([None]),
                    'chi':set([None]),
                    'co':set([None]),
                    'conj':set(['S|S|S','NP|NP|NP','N|N|N','(S|NP)|(S|NP)|(S|NP)']), # don't have schemas yet]), minimal list for now
                    'coord':set(['S|S|S','NP|NP|NP','N|N|N','(S|NP)|(S|NP)|(S|NP)']), # same as above ok for now?
                    'cop':set(['S|NP|NP','S|NP|(N|N)']), # ?
                    'det:art':set(['NP|N']),
                    'det:dem':set(['NP|N']),
                    'det:int':set(['NP']),
                    'det:num':set(['NP|N']),
                    'det:poss':set(['NP|N']),
                    'meta':set([None]),
                    'mod':set([None]),
                    'mod:aux':set([None]),
                    'n':set(['N']),
                    'n:gerund':set(['NP']),
                    'n:let':set(['NP']),
                    'n:prop':set(['NP']),
                    'n:pt':set(['N']), # seems to be for nouns with plural morph.]), like scissors
                    'neg':set(['S|S']), # ?
                    'on':set([None]),
                    'part':set([None]),
                    'poss':set([None]),
                    'post':set(['S|S']), # ?
                    'prep':set(['S|S|NP']),
                    'pro:dem':set(['NP']), # ?
                    'pro:exist':set(['NP']),
                    'pro:indef':set(['NP']),
                    'pro:int':set(['NP']),
                    'pro:obj':set(['NP']),
                    'pro:per':set(['NP']),
                    'pro:poss':set(['NP|N']),
                    'pro:refl':set(['NP']),
                    'pro:rel':set(['NP']),
                    'pro:sub':set(['NP']),
                    'qn':set(['NP|N']),
                    'v':set(['S|NP','S|NP|NP']),
                    #'v:obj':set([None]),
                    'wplay':set([None]),
                    }

pos_marking_dict = {k:set(['X']) if v == set([None]) else v for k,v in pos_marking_dict.items()}

exclude_lfs = [
        'and',
        'att',
        ' _ ',
        '(BARE $1 ($2 (qn|many_2 $1))',
        'lambda $1_{<<e,e>,e>}.lambda $1_{<<e,e>,e>}',
        '{e}.lambda $1_{e}.',
        'v|do-past_1 (v|hit-zero_3 pro:per|you_2 pro:indef|something_4)',
        '$1_{e}.lambda $2_{e}',
        'Q (n:prop|adam_3 pro:per|you_1)',
        'Q (n:prop|ursula_3 pro:per|you_1)',
        'Q (pro:int|what_3 pro:per|you_1)',
        'Q (not (v|hurry_3 you))',
        'Q (n:prop|daddy_3 pro:dem|that_1)',
        'Q (n|right_3 pro:dem|that_2)',
        'n|stop_1 you'
        ]
