
neg_conts = {
                'won\'t': 'wo n\'t',
                'can\'t': 'can n\'t',
                'don\'t': 'do n\'t',
                'didn\'t': 'did n\'t',
                'doesn\'t': 'does n\'t',
                'shouldn\'t': 'should n\'t',
                'couldn\'t': 'could n\'t',
                'wouldn\'t': 'would n\'t',
                'musn\'t': 'mus n\'t',
                'isn\'t': 'is n\'t',
                'wasn\'t': 'was n\'t',
                'aren\'t': 'are n\'t',
                'weren\'t': 'were n\'t',
                'amn\'t': 'am n\'t',
                }

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
        'adv|just_1(BARE($0,det:num|one_2($0)))', # don't know how to model but seems wrong
        'Q(prep|of_2(BARE($0,det:num|two_1(pro:obj|them_3($0)))))', # don't know how to model but seems wrong
        'lambda $0_{r}.n|blow_1(you,BARE($1,pro:indef|one_2($1)),$0)', # unsure how to model but seems wrong
        'lambda $1_{e}.lambda $0_{r}.v|do_1(you,$1,$0)',# unsure how to model but defo wrong
        'lambda $1_{e}.lambda $0_{r}.v|do_1(you,$1,$0)',# unsure how to model but defo wrong
        'lambda $1_{e}.lambda $0_{r}.not(mod|can_2($1(pro:per|you_1,$0),$0),$0)',# unsure how to model but defo wrong
        'lambda $0_{r}.not(n:gerund|break-presp_3(you,$0),$0)',# unsure how to model but defo wrong
        'lambda $0_{r}.Q(cop|be-pres_1(v|finish-past_3(pro:per|you_2,$0),$0))',# unsure how to model but v on finish is wrong
        'lambda $0_{r}.Q(cop|be-pres_1(v|crowd-past_3(pro:per|you_2,$0),$0))',# unsure how to model but v on crowd is wrong
        'lambda $0_{r}.Q(cop|be-3s_1(v|block-past_4(det:art|the_2($1,n|road_3($1)),$0),$0))',# unsure how to model but v on block is wrong
        'BARE($0,det:num|one_1(det:num|two_2(det:num|three_3(det:num|four_4(det:num|five_5($0))))))',# unsure how to model but defo wrong
        'BARE($0,Q(det:num|two_1(v|break_2($0))))',# unsure how to model but defo wrong
        'lambda $0_{r}.Q(adv:int|where_1(qn|all_3(det:art|the_4($1,n|penny-pl_5($1))),$0))',# unsure how to model but defo wrong
        'lambda $0_{r}.not(cop|~be_2(v|eat_5(pro:per|it_1,$0),$0),$0)',# unsure how to model but probs wrong
        'lambda $0_{r}.adv|too_4(adj|just_3(adj|noise-dn_5(pro:per|it_1,$0)))',# unsure how to model but probs wrong
        'lambda $2_{e}.lambda $0_{r}.mod|do-3s_3(v|have_6(n:prop|cromer_5_mr_4,BARE($1,$2(qn|many_2($1))),$0),$0)',# unsure how to model but probs wrong
        'lambda $1_{e}.lambda $0_{r}.v|have_2(v|get-past_4(pro:sub|they_3,$1,$0),$0)',# unsure how to model but probs wrong
        'lambda $0_{r}.not(pro:dem|this_1(pro:per|it_3,n|way_2(pro:per|it_3,$0)),$0)',# unsure how to model but defo wrong
        'not(BARE($0,co|thanks_2($0)))', # wrong and non-semantic anyway
        # Hagar
        'v|ciyēr n|ʔīmaʔ (BARE $1 (n|ʕigūl $1)) adj|niflāʔ' # not well-formed
        'v|racā pro:per|huʔ n|xavitā-BARE' # doesnt match sent
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
        'adj|talk', # malformed
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
    "that 's what he wants", # lf doesn't match
    "you might fall and hurt yourself", # lf doesn't match
    "he had one of those", # lf doesn't match
    "where one what", # lf doesn't match and malformed sent
    "what 's your song about", # lf wrong, treats event-var as wh-var
    "the one who had the bunny rabbit", # lf doesn't match
    'did I knock down more than you did or did you knock more down than I did', # lf doesn't match
    'be very careful with this one', # lf doesn't match
    'are you silly', # lf doesn't match
    'not cranberries', # lf wrong, not sure what it should be
    'two and three what', # lf wrong, not sure what it should be
    'not if you \'re', # lf doesn't match
    'that what honey', # lf doesn't match
    'that \'s not what you said', # lf doesn't match
    "there 's a truck that looks like that", # lf doesn't match
    "you must be need one", # weird and lf defo wrong
    "should be", # weird and lf defo wrong
    "what does it look now", # weird and lf defo wrong
    "you come and look", # weird and lf defo wrong
    "pilot and what", # weird and lf defo wrong
    "you come and look", # weird and lf defo wrong
    "do you remember what this is", # lf doesn't match
    "if I were you", # lf doesn't match
    "I should hope not", # not sure how to model but probs wrong
    "what kind is it", # not sure how to model but probs wrong
    "no wonder what", # not sure how to model but probs wrong
    "where 's the Mommy", # not sure how to model but probs wrong
    "doggie and the car", # not sure how to model but probs wrong
    "put it what", # not sure how to model but wrong
    "did you finish drawing", # not sure how to model but probs wrong
    "what is that you have now", # LF doesn't match
    "that 's something that hasn't developed", # LF doesn't match
    "you drawing a kitty", # LF doesn't match
    "which way it were", # LF doesn't match
    "oh look what you found", # LF doesn't match
    "if he 's not", # LF doesn't match
    "who is this sitting up here", # LF doesn't match
    "I 'll come and do it", # LF doesn't match
    "if you 're", # LF doesn't match
    # Hagar
    'huʔ racā xavitā loʔ melafefōn' # lf doesn't match
    #'ʔābaʔ ʔat rocā' # lf doesn't match--one that Mark says should be wh-movement
    'ʔābaʔ ʔat rocā' # lf doesn't match--one that Mark says should be wh-movement
    ]

direct_take_lf_from_sents = {
    'could be a doctor': 'lambda $0.mod|could (v|equals (det:art|a n|doctor) $0)',
    'you need some what': 'v|need (qn|some n|WHAT) pro:per|you',
    'are those your checkers': 'Q (v|equals (det:poss|your n|checker-pl) pro:dem|those)',
    'they \'re Daddy \'s': 'v|equals n:prop|daddy\'s pro:sub|they',
    'those are David \'s': 'v|equals n:prop|david\'s pro:dem|those',
    'those are David \'s': 'v|equals n:prop|david\'s pro:dem|those',
    'that is Ursula\'s': 'v|equals n:prog|ursula\'s pro:dem|that',
    'ask her what that is': 'lambda $0.v|ask (v|equals pro:int|WHAT pro:dem|that) pro:obj|her $0',
    'ask Ursula what that is': 'lambda $0.v|ask (v|equals pro:int|WHAT pro:dem|that) n:prop|ursula $0',
    'I \'m not hurt': 'not (hasproperty adj|hurt pro:sub|i)',
    'that \'s not rope': 'not (v|equals n|rope-BARE pro:dem|that)',
    'must be a bug': 'lambda $0.mod|must (v|equals (det:art|a n|bug) $0)',
    'it \'s not ready': 'not (hasproperty adj|ready pro:per|it)',
    'that \'s dressing': 'v|equals n|dressing-BARE pro:dem|that',
    'it must be Robin \'s': 'mod|must (v|equals n:prop|robin\'s pro:per|it)',
    'you \'re getting it': 'cop|pres (v|get-prog pro:per|you pro:per|it)',
    'what are you reading': 'Q (cop|pres (v|read-prog pro:per|you pro:int|WHAT))',
    'are you reading a book': 'cop|pres (v|read-prog pro:per|you (det:art|a n|book))',
    'doing tricks': 'lambda $0.v|do-prog n|trick-pl-BARE $0',
    'what happens': 'Q (v|happens pro:int|WHAT)',
    'did you see one': 'Q (mod|do-past (v|see pro:indef|one pro:per|you))',
    'is she finished': 'Q (hasproperty adj|finished pro:sub|she)',
    'what do you call that': 'Q (mod|do (v|call pro:int|WHAT pro:dem|that pro:per|you))',
    'I \'d like some vegetables': 'mod|~genmod (v|like (qn|some n|vegetable-pl) pro:sub|i)',
    'she said two': 'v|say-past pro:indef|two pro:per|she',
    'it is not snow': 'not (v|equals n|snow-BARE pro:per|it)',
    'spear is spear': 'v|equals n|spear-BARE n|spear-BARE',
    'were you lost': 'Q (hasproperty-past adj|lost pro:per|you)',
    'it \'s not round': 'not (hasproperty adj|round pro:per|it)',
    'is n\'t that a surprise': 'Q (not (v|equals (det:art|a n|surprise) pro:dem|that))',
    'what are you drawing': 'Q (cop|pres (v|draw-prog pro:int|WHAT pro:per|you))',
    'that must be what': 'mod|must (v|equals pro:int|WHAT pro:dem|that)',
    'could that be his tail': 'Q (mod|could (v|equals (det:poss|his n|tail) det:dem|that))',
    'that might break': 'mod|might (v|break pro:dem|that)',
    'is n\'t that a rhinoceros': 'Q (not (v|equals (det:art|a n|rhinoceros) pro:dem|that))',
    'is it dry': 'Q (hasproperty adj|dry pro:per|it)',
    "they 're not dry": 'not (hasproperty adj|dry pro:sub|they)',
    "what did we call it": 'Q (mod|do-past (v|call pro:per|it pro:int|WHAT pro:sub|we))',
    "call Robin doctor": 'lambda $0.n|call n:prop|robin n|doctor-BARE $0',
    "is he alright": 'Q (hasproperty adj|alright pro:sub|he)',
    "let go": 'lambda $0.v|let-go $0',
    }

sent_fixes = {
    'an a what': 'and a what',
    'hiʔ ʕoṣā qāqi ʔābaʔ': 'hiʔ ʕoṣā qāqi',
    }

premanual_ida_fixes = {
    'lambda $0_{r}.not(co|careful_5(pro:per|you_2,$0),$0)':'lambda $0_{r}.not(adj|careful_5(pro:per|you_2,$0),$0)', # pos of 'careful' to v
    'lambda $0_{r}.conj|when_5(v|play_7(pro:per|you_6,$0),mod:aux|have_to_2(co|careful_4(pro:per|you_1,$0),$0))':'lambda $0_{r}.conj|when_5(v|play_7(pro:per|you_6,$0),mod:aux|have_to_2(v|careful_4(pro:per|you_1,$0),$0))', # pos of 'careful' to v
    #'lambda $0_{r}.v|need_2(pro:sub|we_1,qn|some_3($1,co|help_4($1)),$0)':'lambda $0_{r}.v|need_2(pro:sub|we_1,qn|some_3($1,n|help_4($1)),$0)', # pos of 'help' to n
    'lambda $0_{r}.Q(det:art|the_3(pro:rel|that_2,n|kitchen_4(pro:rel|that_2,$0)))':'lambda $0_{r}.Q(det:art|the_3(pro:dem|that_2,n|kitchen_4(pro:dem|that_2,$0)))', # pos of 'that' to pro:dem from pro:rel
    'lambda $1_{e}.lambda $0_{r}.$1(co|dum_dum_3,$0)':'lambda $1_{e}.lambda $0_{r}.$1(n:prop|dum_dum_4,$0)', # pos of 'dum_dum' to n:prop from co
    'lambda $1_{e}.lambda $0_{r}.$1(co|dum_dum_3,$0)':'lambda $1_{e}.lambda $0_{r}.$1(n:prop|dum_dum_4,$0)', # pos of 'dum_dum' to n:prop from co
    'lambda $0_{r}.n|stop_1(you,$0)':'lambda $0_{r}.v|stop_1(you,$0)', # pos of 'stop' to v
    'lambda $0_{r}.n|talk_1(you,$0)':'lambda $0_{r}.v|talk_1(you,$0)', # pos of 'talk' to v
    'lambda $0_{r}.Q(n|talk_2(you,$0))':'lambda $0_{r}.Q(v|talk_2(you,$0))', # pos of 'talk' to v
    'lambda $0_{r}.Q(mod|do-3s_1(n|talk_4(det:art|the_2($1,n|microphone_3($1)),$0),$0))':'lambda $0_{r}.Q(mod|do-3s_1(v|talk_4(det:art|the_2($1,n|microphone_3($1)),$0),$0))', # pos of 'talk' to v
    'lambda $0_{r}.Q(mod|can_1(n|talk_3(BARE($1,n|bread_2($1)),$0),$0))':'lambda $0_{r}.Q(mod|can_1(v|talk_3(BARE($1,n|bread_2($1)),$0),$0))', # pos of 'talk' to v
    'lambda $0_{r}.aux|have-3s_2(n:prop|paul_1,det:poss|his_3,$0)':'lambda $0_{r}.v|have-3s_2(n:prop|paul_1,det:poss|his_3,$0)', # pos of 'has' to v
    'lambda $0_{r}.Q(mod|do-3s_1(aux|have_4(det:poss|your_2($1,n|pencil_3($1)),BARE($2,n|number-pl_5($2)),$0),$0))':'lambda $0_{r}.Q(mod|do-3s_1(v|have_4(det:poss|your_2($1,n|pencil_3($1)),BARE($2,n|number-pl_5($2)),$0),$0))', # pos of 'have' to v
    'lambda $0_{r}.mod|~will_2(aux|get_3(pro:sub|i_1,pro:per|you_4,$0),$0)':'lambda $0_{r}.mod|~will_2(v|get_3(pro:sub|i_1,pro:per|you_4,$0),$0)', # pos of 'get' to v
    'lambda $0_{r}.mod|can_2(aux|get_3(pro:per|you_1,pro:per|it_4,$0),$0)':'lambda $0_{r}.mod|can_2(v|get_3(pro:per|you_1,pro:per|it_4,$0),$0)', # pos of 'get' to v
    'lambda $0_{r}.aux|~be_2(n:gerund|get-presp_3(pro:per|you_1,pro:per|it_4,$0),$0)':'lambda $0_{r}.aux|~be_2(v|get-part(pro:per|you_1,pro:per|it_4,$0),$0)', # pos of 'get' to v
    'lambda $0_{r}.aux|have-3s_1(n:prop|adam_2,BARE($1,det:num|two_3(n|pencil-pl_4($1))),$0)':'lambda $0_{r}.v|have-3s_1(n:prop|adam_2,BARE($1,det:num|two_3(n|pencil-pl_4($1))),$0)', # pos of 'have' to v

    # Hagar
    'lambda $0_{r}.Q(v|carīḳ(pro:per|ʔat,BARE($1,on|pīpi($1)),$0))': 'lambda $0_{r}.Q(v|carīḳ(pro:per|ʔat,BARE($1,n|pīpi($1)),$0))', # pos of pipi
    'lambda $0_{r}.and(v|ʕaṣā(BARE($1,on|pīpi($1)),$0),BARE($2,and(adj|gadōl($2),on|pīpi($2))))': 'lambda $0_{r}.and(v|ʕaṣā(BARE($1,on|pīpi($1)),$0),BARE($2,and(adj|gadōl($2),n|pīpi($2))))', # pos of pipi to n
    'lambda $0_{r}.v|ʕaṣā(you,BARE($1,adj|xadāš($1)),$0)': 'lambda $0_{r}.v|ʕaṣā(you,BARE($1,n|xadāš($1)),$0)', # pos of xadas to n
    'lambda $0_{r}.Q(v|racā(pro:per|ʔat,BARE($1,adj|ʔaxēr($1)),$0))': 'lambda $0_{r}.Q(v|racā(pro:per|ʔat,BARE($1,n|ʔaxēr($1)),$0))', # pos of axer to n
    'Q(prep|le(BARE($0,adj|sagōl($0))))': 'Q(prep|le(BARE($0,n|sagōl($0))))', # pos of sagol to n
    'lambda $0_{r}.Q(v|nafāl(BARE($1,adj|cahōv($1)),$0))': 'lambda $0_{r}.Q(v|nafāl(BARE($1,n|cahōv($1)),$0))', # pos of cahov to n
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
    'lambda $0_{r}.v|ciyēr(you,BARE($1,adj|mešulāš($1)),$0)': 'lambda $0_{r}.v|ciyēr(you,BARE($1,n|mešulāš($1)),$0)', # pos of mesulas to n
    'Q (det:art|a (adv|too pro:per|you) (n|pirate (adv|too pro:per|you)))': 'Q (adv|too (cop|be-past pro:per|you (det:art|a n|pirate)))',
    'Q (not (adv|pretty pro:dem|that))': 'Q (not (adj|pretty pro:dem|that))', # pos of 'pretty'
    'Q (pro:int|WHAT det:dem|those)': 'Q (v|equals pro:dem|this pro:int|WHAT)', # pos of 'pretty'
    #'lambda $0.part|play-presp (det:art|the n|piano) $0': 'lambda $0.v|play-prog (det:art|the n|piano) $0', # make prog
    "not (n:prop|daddy's' (hasproperty pro:dem|that adj|suitcase) pro:dem|that)": "not (equals (n:prop|daddy's' n|suitcase) pro:dem|that)", # fix hasproperty
    'Q (cop|pres (v|take-prog pro:int|WHAT pro:per|you))': 'Q (cop|pres (v|take-prog (det:art|the n|WHAT) pro:per|you))', # make 'what' noun
    'lambda $0.Q (cop|pres-3s (v|rain-prog $0))': 'Q (cop|pres-3s (v|rain-prog pro:per|it))', # remove lmbda
    'v|want pro:int|WHAT pro:per|you': 'v|want (det:num|one n|WHAT) pro:per|you', # remove lmbda
    'cop|be-3s pro:per|it': 'v|be pro:per|it', # v|be
    'cop|be-pres pro:sub|they': 'v|be pro:sub|they', # v|be
    'cop|be-pres (qn|any n|propel-dv)': 'v|be (qn|any n|propel-dv)', # v|be
    'lambda $0.cop|become (det:art|a n|spider) $0': 'lambda $0.v|become (det:art|a n|spider) $0', # v|become
    'Q (mod|do (adj|mean pro:int|WHAT pro:per|you))': 'Q (mod|do (v|mean pro:int|WHAT pro:per|you))', # v|mean
    'not (mod|do-past pro:sub|i)': 'not (v|do pro:sub|i)', # v|do
    #'cop|~be (det:poss|your n|pencil)': 'v|exist (det:poss|your n|pencil)', # v|exists
    'cop|be-pres pro:per|you': 'v|be pro:per|you', # v|be
    'lambda $0.n|push n|tire-BARE $0': 'lambda $0.v|push n|tire-BARE $0', # v|push
    'cop|be-3s (det:art|a n|WHAT)': 'lambda $0.v|equals (det:art|a n|WHAT) $0',
    'prep|about pro:dem|this n:prop|utah': 'v|about pro:dem|this n:prop|utah',
    'cop|look-past pro:sub|i': 'v|look-past pro:sub|i', # v|look
    'cop|look-3s n|same-BARE pro:per|it': 'v|look (det:art|the n|same) pro:per|it', # v|look
    'cop|be-pres pro:sub|we': 'v|be pro:sub|we', # v|be
    'cop|be-past pro:per|you': 'v|be pro:per|you', # v|be
    'mod|do-past pro:sub|i': 'v|do pro:sub|i', # v|do
    'mod|do-past pro:per|you': 'v|do pro:per|you', # v|do
    'v|gas pro:dem|this': 'v|equals n|gas-BARE pro:dem|this', # v|equals
    'Q (mod|do-3s (n|spell pro:int|WHAT pro:dem|that))': 'Q (mod|do-3s (v|spell pro:int|WHAT pro:dem|that))', # v|spell
    'not (mod|do (adj|mean n|ball-BARE pro:per|you))': 'not (mod|do (v|mean n|ball-BARE pro:per|you))', # v|mean
    'Q (mod|do-past (prep|like pro:per|it pro:per|you))': 'Q (mod|do-past (v|like pro:per|it pro:per|you))', # v|like
    'Q (pro:int|WHAT pro:per|it)': 'Q (v|equals pro:int|WHAT pro:per|it)', # always missing equals fsr
    'not (mod|will (v|frighten pro:per|you pro:sub|she))': 'not (mod|will (v|frighten pro:per|you pro:sub|she))', # always missing equals fsr
    "n|boy's' (hasproperty pro:dem|that adj|hat) pro:dem|that": "v|equals (det:art|a (adj|boy's' n|hat)) pro:dem|that", # always missing equals fsr
    'not (v|cool pro:per|it)': 'not (hasproperty adj|cool pro:per|it)',
    'Q (mod|~genmod (v|say pro:int|WHAT pro:sub|i))': 'Q (mod|do-past (v|say pro:int|WHAT pro:sub|i))',
    'Q (n|bit (det:poss|your n|pants) pro:int|WHO)': 'Q (v|bite-past (det:poss|your n|pants) pro:int|WHO)',
    'hasproperty pro:dem|this adj|ice+cream': 'v|equals n|ice-BARE pro:dem|this',
    'Q (mod|do-past (n|miss pro:indef|one pro:per|you))': 'Q (mod|do-past (v|miss pro:indef|one pro:per|you))',
    'v|have-3s det:poss|his n:prop|paul': 'v|have-3s pro:poss|his n:prop|paul',
    #'cop|~be (qn|another n|story)': 'v|exist (qn|another n|story)', # v|exist
    # Hagar
    'v|ciyēr pro:per|ʔat n|ʕigūl-BARE': 'v|ciyēr you n|ʕigūl-BARE', # split anything into two words like it is in the sent (which may well be mistranscribed)
    }

manual_sent_fixes = {
    "'s busy": "Adam 's busy", # my fault, my preproc removes all 'Adam's
    "is a clown": "Adam is a clown", # my fault, my preproc removes all 'Adam's
    "because you spilled it": "you spilled it", # to match lf
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
