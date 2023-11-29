pos_marking_dict = {
                    'adj':set(['N|N']),
                    'adv':set([None]),
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
                    'prep':set(['S|NP|NP', 'S|S|NP']),
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

pos_marking_dict = {k:set(['X']) if v == set([None]) else v for k,v in pos_marking_dict.items()}


base_lexicon = {k:set(['NP']) for k in ('you','i','me','he','she','it','WHAT','WHO')}
base_lexicon['equals'] = set(['S|NP|NP'])
base_lexicon['hasproperty'] = set(['S|NP|(N|N)'])
base_lexicon['mod|will_2'] = set(['S|NP', 'S|NP|NP'])
base_lexicon['not'] = set(['X'])

full_lfs_to_exclude = [
        # Adam
        'lambda $0_{r}.cop|be-pres_1(part|write-presp_3(pro:per|you_2,det:art|a_4($1,n|letter_5($1)),$0),$0)',
        'n|stop_1 you',
        'n|talk_1 you',
        'Q (n:prop|adam_3 pro:per|you_1)',
        'Q (n:prop|ursula_3 pro:per|you_1)',
        'Q (pro:int|what_3 pro:per|you_1)',
        'Q (not (v|hurry_3 you))',
        'Q (n:prop|daddy_3 pro:dem|that_1)',
        'Q (n|right_3 pro:dem|that_2)',
        'cop|be-pres_1 (part|write-presp_3 pro:per|you_2 (det:art|a_4 n|letter_5))',
        'v|have_2 (v|get-past_4 pro:sub|they_3 WHAT)',
        'not (v|know_4 pro:sub|he_1 (BARE n|boy-pl_7))',
        'BARE $0 (pro:indef|more_2 $0)', # not well-formed
        'det:art|the_1 $0 (pro:indef|one_2 $0)', # not well-formed
        'adv|enough_3', # not well-formed
        'v|have_2 pro:per|you_1 (det:art|a_3 n|funnel_4)', # doesn't match sent
        'v|have pro:per|you (det:art|a n|funnel)', # doesn't match sent
        'v|open pro:per|you pro:per|it', # doesn't match sent
        'Q (prep|like_1 pro:dem|that_2)', # just prepositional phrase
        'qn|some n|water', # just noun phrase
        'n|tickle WHO pro:obj|me', # 'tickle' pos wrong and sent is malordered
        'n|call you n:prop|robin', # 'call' pos wrong and not clear what sent means
        # Hagar
        'v|ciyēr n|ʔīmaʔ (BARE $1 (n|ʕigūl $1)) adj|niflāʔ' # not well-formed
        ]

partial_lfs_to_exclude = [
        # Adam
        'and',
        'att',
        ' _ ',
        '(BARE $1 ($2 (qn|many_2 $1))',
        'lambda $1_{<<e,e>,e>}.lambda $1_{<<e,e>,e>}',
        '{e}.lambda $1_{e}.',
        'v|do-past_1 (v|hit-zero_3 pro:per|you_2 pro:indef|something_4)',
        '$1_{e}.lambda $2_{e}',
        '(lambda $1_{r}.v|swallow_5 pro:per|you_2 pro:per|it_6 $1)',
        'pro:per|it_3_at_2',
        '(lambda $1_{r}.', # means embedded S and haven't yet removed the event vars from them
        ',lambda $1_{r}', # means embedded S and haven't yet removed the event vars from them
        'BARE($0,pro:indef|something_1($0))', # sent is wrong
        # Hagar
        'qn|gam', # wrong pos
        'qn|raq', # wrong pos
        'sing|', # meaningless pos
        'chi|', # meaningless pos
        'pro:per|štey', # wrong pos
        'pro:per|kol', # wrong pos
        'co|loʔ($1,n|', # mistaken use of co| as determiner
        'co|loʔ($0,n|', # mistaken use of co| as determiner
        'co|zēhu($0,n|', # mistaken use of co| as determiner
        'co|maspīQ($1,n|', # mistaken use of co| as determiner
        ]

he_chars = "qhvSWl~N?ʔmetṭ   טʕJ!.FbcM_kçGDHwxByQẒLKAgYRPTnoֿ+=CzpšXṣʝfi&d́:arusū"
# he_chars = ''.join(pd.read_csv('hebrew_latin_table.tsv',sep='\t')['Unnamed: 0'])
exclude_lfs = full_lfs_to_exclude + partial_lfs_to_exclude

manual_ida_fixes = { # applied after conversion to no-comma form
    'Q (v|do-past (n|miss_3 pro:per|you pro:indef|one))': 'Q (v|do-past (v|miss pro:per|you pro:indef|one))', # pos of 'miss' to verb
    'n|stop you pro:dem|that': 'v|stop you pro:dem|that', # pos of 'stop' to verb
    'Q (v|do-past (prep|like pro:per|you pro:per|it))': 'Q (v|do-past (v|like pro:per|you pro:per|it))', # pos of 'like' to verb
    'n|call you n:prop|daddy': 'v|call you n:prop|daddy', # pos of 'call' to verb
    'equals pro:per|you (det:art|the n|drive-dv)': 'Q (equals pro:per|you (det:art|the n|drive-dv))', # add 'Q'
    'hasproperty you n|stop': 'v|stop you', # add 'Q'
    'n|stop you pro:dem|that': 'v|stop you pro:dem|that', # pos of 'stop' to verb
    'n|stop you pro:rel|that': 'v|stop you pro:rel|that', # pos of 'stop' to verb
    }
