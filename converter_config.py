full_lfs_to_exclude = [
        # Adam
        'lambda $0_{r}.cop|be-pres_1(part|write-presp_3(pro:per|you_2,det:art|a_4($1,n|letter_5($1)),$0),$0)',
        'lambda $1_{e}.lambda $0_{r}.$1(qn|many_3(BARE($2,n|bear-pl_4($2)),$0))', # superfluous 'BARE'
        'lambda $0_{r}.v|say-3s_2(n|mine_1,BARE($1,n:let|p_3($1)),$0)', # doesn't match sent
        'lambda $0_{r}.v|have_2(pro:per|you_1,det:art|a_3($1,n|funnel_4($1)),$0)', # doesn't match sent
        'lambda $0_{r}.co|thank_you_1(you,$0)', # unclear semantics, unclear semcat
        'lambda $0_{r}.Q(det:art|a_3(pro:rel|that_2,and_comp(n|dragon_4(pro:rel|that_2,$0),n|fly_5(pro:rel|that_2,$0))))', # malformed
        'lambda $1_{e}.lambda $0_{r}.$1(co|dum_dum_4,$0)', # dum_dum wrong pos and no 'or' as in sent
        'lambda $1_{e}.lambda $0_{r}.$1(qn|many_3(BARE($2,n|bear-pl_4($2)),$0))', # not well-formed
        'lambda $0_{r}.mod|~will_2(adj|soon_5(adj|ready_4(pro:per|it_1,$0)),$0)', # don't know how to model
        'lambda $0_{r}.cop|be-3s_3_there_1_it_2(you,$0)', # don't know how to model but seems wrong
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
        'co|oh_1_no_2', # malformed
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

exclude_sents = [
    "he doesn 't know dogs and boys", # lf doesn't match
    'this is the man who drives the busy bulldozer', # lf doesn't match
    'that \'s the man who wrote the book', # lf doesn't match
    'that \'s no the one that has the markings on it', # lf doesn't match
    #'that \'s the stool I use when I wash dishes', # lf doesn't match
    #'Shadow Gay is the horse that won the Kentucky Derby', # lf doesn't match
    'the part that you open so_that you can pull the kleenex up through there', # lf doesn't match
    "do you have a tractor that s smaller than that one", # lf doesn't match
    "those are Cinderella s sisters trying the slipper on", # lf doesn't match
    'this is the cheese Adam always picks out at the grocery store', # lf doesn't match
    'Adam whose racket is that', # not sure how to model wh-possessive
    'don \'t Adam foot',
    'who is going to become a spider',
    'two Adam',
    'a d a m'
    "he doesn 't know dogs and boys", # lf doesn't match
    "see the eggs the milk the butter", # lf doesn't match
    "it 's the cover that came off the puzzle", # lf doesn't match
    'Shadow_Gay is the horse that won the Kentucky Derby', # lf doesn't match
    "that 's the stool I use when I wash dishes", # lf doesn't match
    "one of the members of George Washington 's army", # lf doesn't match
    'I see some pennies that you dropped from your pocket', # lf doesn't match
    "that 's because I had a cold but I don 't any more", # lf doesn't match
    "do you have a tractor that 's smaller than that one", # lf doesn't match
    "those are Cinderella 's sisters trying the slipper on", # lf doesn't match
    "the man who 's standing up has a guitar", # lf doesn't match
    'this is the cheese always picks out at the grocery store', # lf doesn't match
    'did you like the balloon that you blow up', # lf doesn't match
    "there 's a dot that says cross your printing set", # lf doesn't match
    "but it 's something that goes along with the train", # lf doesn't match
    'for the cars that are waiting to cross the track maybe', # lf doesn't match
    "no it 's more fun when it 's bigger", # lf doesn't match
    'in the refrigerator so it will be nice and warm', # lf doesn't match
    "I don 't want any butter that might fall off the truck", # lf doesn't match
    'did you show Urs your name embroidered on your sunsuit', # lf doesn't match
    'I see two shapes that look just like that', # lf doesn't match
    "isn 't that a shirt that 's hanging on the line", # lf doesn't match
    "wouldn 't know what to do with an airplane", # lf doesn't match
    'can you find a key that fits the lock', # lf doesn't match
    "I 'm breaking the stick that the motor of I made", # lf doesn't match
    'did you tell Ursula what happened in the barber shop', # lf doesn't match
    'this is a q with the little thing sticking out', # lf doesn't match
    'the one you were singing at the table', # lf doesn't match
    "don 't you have something to show her", # lf doesn't match
    "no more", # lf doesn't match
    "now no more", # lf doesn't match
    "no d ", # lf doesn't match
    "where is the a", # lf doesn't match
    "isn 't that what your Daddy does", # lf doesn't match
    "they 're not the kind that you eat", # lf doesn't match
    "you have some thing to show me", # lf doesn't match
    "the kind the policeman carry", # lf doesn't match
    "because that 's what he wants", # lf doesn't match
    "because you might fall and hurt yourself", # lf doesn't match
    ]

sent_fixes = {
    'hiʔ ʕoṣā qāqi ʔābaʔ': 'hiʔ ʕoṣā qāqi',
    }

premanual_ida_fixes = {
    'lambda $0_{r}.not(co|careful_5(pro:per|you_2,$0),$0)':'lambda $0_{r}.not(adj|careful_5(pro:per|you_2,$0),$0)', # pos of 'careful' to v
    'lambda $0_{r}.conj|when_5(v|play_7(pro:per|you_6,$0),mod:aux|have_to_2(co|careful_4(pro:per|you_1,$0),$0))':'lambda $0_{r}.conj|when_5(v|play_7(pro:per|you_6,$0),mod:aux|have_to_2(v|careful_4(pro:per|you_1,$0),$0))', # pos of 'careful' to v
    'lambda $0_{r}.v|need_2(pro:sub|we_1,qn|some_3($1,co|help_4($1)),$0)':'lambda $0_{r}.v|need_2(pro:sub|we_1,qn|some_3($1,n|help_4($1)),$0)', # pos of 'help' to n
    'lambda $0_{r}.Q(det:art|the_3(pro:rel|that_2,n|kitchen_4(pro:rel|that_2,$0)))':'lambda $0_{r}.Q(det:art|the_3(pro:dem|that_2,n|kitchen_4(pro:dem|that_2,$0)))', # pos of 'that' to pro:dem from pro:rel
    'lambda $1_{e}.lambda $0_{r}.$1(co|dum_dum_3,$0)':'lambda $1_{e}.lambda $0_{r}.$1(n:prop|dum_dum_4,$0)', # pos of 'dum_dum' to n:prop from co
    'lambda $1_{e}.lambda $0_{r}.$1(co|dum_dum_3,$0)':'lambda $1_{e}.lambda $0_{r}.$1(n:prop|dum_dum_4,$0)', # pos of 'dum_dum' to n:prop from co
    }

manual_ida_fixes = { # applied after conversion to no-comma form
    'Q (v|do-past (n|miss_3 pro:per|you pro:indef|one))': 'Q (v|do-past (v|miss pro:per|you pro:indef|one))', # pos of 'miss' to verb
    'n|stop you pro:dem|that': 'v|stop you pro:dem|that', # pos of 'stop' to verb
    'Q (v|do-past (prep|like pro:per|you pro:per|it))': 'Q (v|do-past (v|like pro:per|you pro:per|it))', # pos of 'like' to verb
    'n|call you n:prop|daddy': 'v|call you n:prop|daddy', # pos of 'call' to verb
    'equals pro:per|you (det:art|the n|drive-dv)': 'Q (equals pro:per|you (det:art|the n|drive-dv))', # add 'Q'
    'hasproperty you n|stop': 'v|stop you', # add 'Q'
    'n|stop you pro:dem|that': 'v|stop you pro:dem|that', # pos of 'stop' to verb
    'n|stop you pro:rel|that': 'v|stop you pro:rel|that', # pos of 'stop' to verb
    'cop|be-3s_3_there_1_it you': 'v|equals', # pos of 'stop' to verb
    'Q (mod|do-past (n|miss pro:per|you pro:indef|one))': 'Q (mod|do-past (v|miss pro:per|you pro:indef|one))', # pos of 'miss' to verb
    'Q (mod|will-cond (conj|like pro:per|you (det:art|a n|piece)))': 'Q (mod|will-cond (v|like pro:per|you (det:art|a n|piece)))', # pos of 'like' to verb
    'qn|another $0 (pro:indef|one $0)': 'qn|another pro:indef|one', # missed 'one' as noun
    'Q (mod|do (v|see pro:per|you n|thing_5_any))': 'Q (mod|do (v|see pro:per|you (qn|any n|thing)))', # split anything into two words like it is in the sent (which may well be mistranscribed)
    'Q (pro:dem|that $1 (pro:indef|one $1))': 'Q (pro:dem|that $1 (pro:indef|one $1))', # split anything into two words like it is in the sent (which may well be mistranscribed)
    'v|ciyēr pro:per|ʔat n|ʕigūl-BARE': 'v|ciyēr you n|ʕigūl-BARE', # split anything into two words like it is in the sent (which may well be mistranscribed)
    }

manual_sent_fixes = {
    "'s busy": "Adam 's busy", # my fault, my preproc removes all 'Adam's
    "is a clown": "Adam is a clown", # my fault, my preproc removes all 'Adam's
    }

hagar_svo_sentences = {
    "ʔat mecayēret ʕigūl", # Line 649
    "ʔanāxnu loxcōt yadāyim", # Line 833
    "ʔat roʔā galīm", # Line 1065
    "ʔat qanīt matanā le sābaʔ", # Line 1645
    "ʔanī mexapēṣet pitarōn le ~ha xidā ha qašē ha zōʔti", # Line 3165
    "Bindī ʔatā ʔohēv salāṭ", # Line 6433
    "ʔanī ʔoḳāl ʔavoqādo bli limōn we mēlax", # Line 6533
    "ʔatā rocē glīda", # Line 17277
    "ʔat kotēvet ʔat kotēvet miḳtāv", # Line 19129
    "ʔat ṣāma sukār", # Line 23829
}
