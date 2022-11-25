def sample(entry, sentence_chart, rule_set):
    """Tom's code, supposed to return the most likely parse after the
    inside_outside chart has been filled out.
    """
    children = []
    rule_score = rule_set.return_log_prob(entry.syn_key, entry.syn_key+'_LEX')
    children.append((entry.word_score+entry.sem_score+rule_score, 'LEX'))
    for pair in entry.children:
        left_syn = pair[0][0]
        right_syn = pair[1][0]
        target = left_syn+'#####'+right_syn
        try:
            children.append((rule_set.return_log_prob(entry.syn_key, target) +
                     sentence_chart[pair[0][3]-pair[0][2]][pair[0]].inside_score +
                     sentence_chart[pair[1][3]-pair[1][2]][pair[1]].inside_score, pair))
        except KeyError:
            print(pair[0], 'not found in sentence_chart')
    children = sorted(children,key=lambda x:x[0], reverse=True)
    if children[0][1] == 'LEX':
        return [(entry.word_target, entry.syn_key, entry.sem_key)]
    else:
        best_parse = children[0]
        pair = best_parse[1] # two elements that can best combine to give entry
        left_syn = pair[0][0] # syntactic category of the left element
        right_syn = pair[1][0] # syntactic category of the right element
        target = left_syn+'#####'+right_syn # unused?
        pl = sample(sentence_chart[pair[0][3]-pair[0][2]][pair[0]], sentence_chart, rule_set)
        pr = sample(sentence_chart[pair[1][3]-pair[1][2]][pair[1]], sentence_chart, rule_set)
        pl.extend(pr)
        return pl
