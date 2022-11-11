# this is a function to sample the most probable parse for 
# each of the training sentences once the inside_outside
# algorithm has converged
"""
def sample_most_probable_parse(sentence_charts,RuleSet,lexicon,output,sem_store):
    for c in sentence_charts:
        # first of all build up inside probabilities
        for level in sentence_charts[c]:
            for item in sentence_charts[c][level]:
                entry = sentence_charts[c][level][item]
                entry.word_prob = lexicon.get_word_prob(entry.word_target,entry.cat.key,entry.sem_key)
                entry.sem_prob = lexicon.get_sem_prob(entry.cat.key,entry.sem_key,sem_store)
                entry.inside_prob =  entry.word_prob*entry.sem_prob*RuleSet.return_prob(entry.cat.key,entry.cat.key+'_LEX')
                rule_prob = 0
                for pair in entry.children:
                    left_syn = pair[0][0]
                    right_syn = pair[1][0]
                    target = left_syn+'#####'+right_syn
                    entry.inside_prob = entry.inside_prob + RuleSet.return_prob(entry.cat.key,target)*sentence_charts[c][pair[0][3]-pair[0][2]][pair[0]].inside_prob*sentence_charts[c][pair[1][3]-pair[1][2]][pair[1]].inside_prob
                    rule_prob += RuleSet.return_prob(entry.cat.key,target)
                if entry.inside_prob > 1.0:
                    print 'inside prob is over one'
                    print item
                    print 'rule prob is ',rule_prob
                    
        for item in  sentence_charts[c][len(sentence_charts[c])]:
            top = sentence_charts[c][len(sentence_charts[c])][item]
            parse = sample(top,sentence_charts[c],RuleSet)
            print >> output,parse
"""


def sample(entry,sentence_chart,RuleSet):
    """
    Tom's code. It is supposed to return the most likely parse after the inside_outside chart has
    been filled out.
    """
    children = []
    children.append((entry.word_score+entry.sem_score+\
                         RuleSet.return_log_prob(entry.syn_key,entry.syn_key+'_LEX'),'LEX'))
    for pair in entry.children:
        left_syn = pair[0][0]
        right_syn = pair[1][0]
        target = left_syn+'#####'+right_syn
        children.append((RuleSet.return_log_prob(entry.syn_key,target) + \
                     sentence_chart[pair[0][3]-pair[0][2]][pair[0]].inside_score + \
                     sentence_chart[pair[1][3]-pair[1][2]][pair[1]].inside_score,pair))
    children.sort()
    children.reverse()
    if children[0][1] == 'LEX':
        return [(entry.word_target,entry.syn_key,entry.sem_key)]
    else:
        pair = children[0][1]
        left_syn = pair[0][0]
        right_syn = pair[1][0]
        target = left_syn+'#####'+right_syn
        pl = sample(sentence_chart[pair[0][3]-pair[0][2]][pair[0]],sentence_chart,RuleSet)
        pr = sample(sentence_chart[pair[1][3]-pair[1][2]][pair[1]],sentence_chart,RuleSet)
        pl.extend(pr)
        return pl
