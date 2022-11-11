import sys, pylab, pdb, os
import numpy as np
import matplotlib.pyplot as plt
import extract_from_lexicon3

# if len(sys.argv) != 2:
#    print('Usage: eval_test_file.py ')
#    sys.exit(-1)

FIELD_NAMES = ['name', 'sentence count', 'row type', 'LF', 'word', 'prob']


def draw_lf_given_w(phenom, words, filenames):
    results = dict([(w, []) for w in words])
    seen_sentence_counts = set([0])
    for fn in filenames:
        f = open(fn)
        for line in f:
            line = line.strip()
            if "Pr(correct LF|w)" in line:
                fields = dict(zip(FIELD_NAMES, line.split('\t')))
                seen_sentence_counts.update([int(fields['sentence count'])])
                if fields['word'] in words and phenom.target_lf(fields['word']) == fields['LF']:
                    entry = results[fields['word']]
                    entry.append((int(fields['sentence count']), float(fields['prob'])))
                    results[fields['word']] = entry
    for w, res in results.items():
        if res == []:
            del results[w]
        else:
            cur_sentence_counts = set([int(r[0]) for r in res])
            for sc in seen_sentence_counts - cur_sentence_counts:
                res.append((sc, 0.0))
            results[w] = res
    return results


def draw_syn_cat(target_syn_cats, filenames):
    results = dict([(sc, []) for sc in target_syn_cats])
    seen_sentence_counts = set()
    for fn in filenames:
        f = open(fn)
        for line in f:
            line = line.strip()
            if "Pr(syn|all relevant syns)" in line:
                fields = dict(zip(FIELD_NAMES, line.split('\t')))
                seen_sentence_counts.update([int(fields['sentence count'])])
                if fields['syn_cat'] in target_syn_cats:
                    entry = results[fields['syn_cat']]
                    entry.append((int(fields['sentence count']), float(fields['prob'])))
                    results[fields['syn_cat']] = entry
    for syncat, res in results.items():
        cur_sentence_counts = set([int(r[0]) for r in res])
        for sc in seen_sentence_counts - cur_sentence_counts:
            res.append((sc, 0.0))
        results[syncat] = res
    return results


def draw_graph(results):
    legend = []
    for w, res in results.items():
        arr = np.array(res)
        arr = arr[arr[:, 0].argsort(),]
        plt.plot(arr[:, 0], arr[:, 1], linewidth=3.0)
        legend.append(w)

    plt.legend(legend, loc='lower right')
    plt.show()


if len(sys.argv) == 1:
    print('Usage: draw_lf_given_w_graph.py <directory for files> [<suffix for files (: delimited)>]')
    sys.exit(-1)

if len(sys.argv) == 3:
    suffixes = sys.argv[2].split(':')
    filenames = []
    for suffix in suffixes:
        filenames.append([sys.argv[1] + '/' + fn for fn in os.listdir(sys.argv[1]) if fn.endswith(suffix)])
else:
    suffixes = ['All']
    filenames = [[sys.argv[1] + '/' + fn for fn in os.listdir(sys.argv[1])]]

"""
# getting nouns
word_lfs = extract_from_lexicon3.get_noun_lfs()
phenom = extract_from_lexicon3.Phenomenon('Nouns', word_lfs,'N',['N'])
words = ['girl', 'coat', 'table', 'coffee', 'pencil', 'spoon']
results = draw_lf_given_w(phenom,words,filenames)
draw_graph(results)

# getting prepositions
word_lfs = extract_from_lexicon3.get_prep_lfs()
phenom = extract_from_lexicon3.Phenomenon('Prepositions', word_lfs, '(PP/NP)', \
                                     ['(PP/NP)', "(PP\\NP)"])
words = ['in','on','by','about']
results = draw_lf_given_w(phenom,words,filenames)
draw_graph(results)

# getting determiners
word_lfs = extract_from_lexicon3.get_det_lfs()
phenom = extract_from_lexicon3.Phenomenon('Determiners', word_lfs, '(NP/N)', \
                                              ['(NP/N)', "(NP\N)"])
words = ['the','a','any','his']
results = draw_lf_given_w(phenom,words,filenames)
draw_graph(results)

# daxed
phenom = extract_from_lexicon3.get_dax_phenom()
word_lfs = [('daxed', 'lambda $0_{e}.lambda $1_{e}.lambda $2_{ev}.v|dax&PAST($1,$0,$2)'),\
                ('dax', 'lambda $0_{e}.lambda $1_{e}.lambda $2_{ev}.v|dax($1,$0,$2)')]
phenom = extract_from_lexicon3.Phenomenon('Daxed', word_lfs, '((S\\NP)/NP)',\
                          ['((S\\NP)/NP)', "((S/NP)/NP)", "((S\\NP)\\NP)", "((S/NP)\\NP)"],\
                          ['daxed','dax'])
words = ['daxed','dax']
results = {}
for fns, suffix in zip(filenames,suffixes):
    res = draw_lf_given_w(phenom,words,fns)
    for w,r in res.items():
        results[(suffix,w)] = r
"""

phenom = extract_from_lexicon3.get_jax_phenom()
words = ['jax']
results = {}
for fns, suffix in zip(filenames, suffixes):
    res = draw_lf_given_w(phenom, words, fns)
    for w, r in res.items():
        results[(suffix, w)] = r

print(results)
draw_graph(results)
