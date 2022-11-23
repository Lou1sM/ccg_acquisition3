# this is to generate a sentence from a logical expression.
from build_inside_outside_chart import build_chart
from inside_outside_calc import i_o_oneChart
from sample_most_probable_parse import sample
from math import exp

def assignWords(chart, lexicon):
    for level in chart:
        for item in chart[level]:
            entry = chart[level][item]
            ccgcat = entry.ccgCat
            w_syn_sem = lexicon.getMaxWordForSynSem(ccgcat.syn.to_string(), ccgcat.sem.to_string(True))
            if w_syn_sem is None: continue
            entry.words = w_syn_sem[0]
            entry.word_target = w_syn_sem[0]

# can't just sample down since we need to generate the correct semantics
def generateSent(lexicon, rule_set, top_cat, cat_store, sem_store, is_exclude_mwe, corrSent, genoutfile, sentence_count, sentnum):
    # pack 'word list' with None so that we can reuse old code
    # generate words at leaves
    # do MAP inside-outside
    wordlist = ["placeholderW"]*(len(top_cat.sem.allSubExps()))
    chart = build_chart([top_cat], wordlist, rule_set, lexicon, cat_store, sem_store, is_exclude_mwe)
    assignWords(chart, lexicon)
    i_o_oneChart(chart, sem_store, lexicon, rule_set, False, 0.0, lexicon.sentence_count, True)
    topparses = []
    for entry in chart[len(chart)]:
        top = chart[len(chart)][entry]
        topparses.append((top.inside_score, top))

    top_parse = sample(sorted(topparses)[-1][1], chart, rule_set)
    print("\ntop generated parse"+str(sentnum)+":", file=genoutfile)
    print(top_parse, file=genoutfile)
    print(top.inside_score, file=genoutfile)

    print("\ntop generated parse"+str(sentnum)+":")
    print(top_parse)
    print(top.inside_score)

    chart = build_chart([top_cat], corrSent.split(), rule_set, lexicon, cat_store, sem_store, is_exclude_mwe)
    if chart is not None:
        corr_score = i_o_oneChart(chart, sem_store, lexicon, rule_set, False, 0.0, lexicon.sentence_count)
        print("corr score"+str(sentnum)+" is ", corr_score, file=genoutfile)
        print("prob gen corr"+str(sentnum)+" = ", exp(corr_score - top.inside_score), file=genoutfile)
        print("prob gen corr"+str(sentnum)+" = ", exp(corr_score - top.inside_score))
