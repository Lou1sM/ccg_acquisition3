import sys, pylab, pdb, os
import numpy as np
import matplotlib.pyplot as plt
import extract_from_lexicon
from optparse import OptionParser

# if len(sys.argv) != 2:
#    print('Usage: eval_test_file.py ')
#    sys.exit(-1)

ALL_SUFFIXES = 'ALL_SUFFIXES'
FIELD_NAMES_LF_GIVEN_W = ['name', 'sentence count', 'row type', 'LF', 'word', 'prob']
FIELD_NAMES_SYN = ['name', 'sentence count', 'syn_cat', 'row_type', 'prob']


def draw_lf_given_w(phenom, words, filenames, field_names):
    results = dict([(w, []) for w in words])
    seen_sentence_counts = set([0])
    for fn in filenames:
        f = open(fn)
        for line in f:
            line = line.strip()
            if "Pr(correct LF|w)" in line:
                fields = dict(zip(field_names, line.split('\t')))
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


def draw_syn_cats(target_syn_cats, filenames, field_names):
    results = dict([(sc, []) for sc in target_syn_cats])
    seen_sentence_counts = set()
    for fn in filenames:
        f = open(fn)
        for line in f:
            line = line.strip()
            if "Pr(syn|all relevant syns)" in line:
                fields = dict(zip(field_names, line.split('\t')))
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


def draw_graph(results, options):
    legend = []
    for w, res in results.items():
        arr = np.array(res)
        arr = arr[arr[:, 0].argsort(),]
        plt.plot(arr[:, 0], arr[:, 1], linewidth=3.0)
        if w[0] == ALL_SUFFIXES:
            legend.append(w[1])
        elif w[0] == 'out2':
            legend.append(w[1] + ' pronoun args')
        elif w[0] == 'out8':
            legend.append(w[1] + ' noun args')
        elif w[0] == 'out11':
            legend.append(w[1] + ' NP+PP')
        elif w[0] == 'out5':
            legend.append('dax example #1')
        elif w[0] == 'out6':
            legend.append('dax example #2')
        else:
            legend.append(w)
    plt.ylim(-0.1, 1.1)
    # plt.xlim(0,800)
    if not options.nolegend:
        plt.legend(legend, loc='lower right')
    plt.show()


def cmd_line_parser():
    """
    Returns the command line parser.
    """
    usage = "usage: %prog [options]\n"
    opt_parser = OptionParser(usage=usage)
    opt_parser.add_option("-d", action="store", dest="direc",
                          help="directory to take files from")
    opt_parser.add_option("-w", action="store", dest="words", default='',
                          help="the words to inspect, : delimited")
    opt_parser.add_option("-s", action="store", dest="suffixes",
                          help="the suffixes for the files")
    opt_parser.add_option("-p", action="store", dest="phenom_type",
                          help="the phenomenon to inspect")
    opt_parser.add_option("--nolegend", action="store_true", dest="nolegend",
                          help="omit the legend")
    opt_parser.add_option("-c", dest='infix', action="store", default='',
                          help="infix to be included in the filenames taken into account")
    return opt_parser


####################
# MAIN
####################

if __name__ == '__main__':
    parser = cmd_line_parser()
    options, args = parser.parse_args(sys.argv)

    direc = options.direc
    words = options.words.split(':')
    if options.suffixes:
        suffixes = options.suffixes.split(':')
        filenames = []
        for suffix in suffixes:
            filenames.append(
                [direc + '/' + fn for fn in os.listdir(direc) if fn.endswith(suffix) and options.infix in fn])
    else:
        suffixes = [ALL_SUFFIXES]
        filenames = [[direc + '/' + fn for fn in os.listdir(direc) if options.infix in fn]]

    if options.phenom_type == 'trans':
        word_lfs = extract_from_lexicon.get_transitive_lfs()
        phenom = extract_from_lexicon.Phenomenon('Transitives', word_lfs, '((S\\NP)/NP)', \
                                                  ["((S\\NP)/NP)", "((S/NP)/NP)", "((S\\NP)\\NP)", "((S/NP)\\NP)"], \
                                                  two_LFs=True, \
                                                  flipped_target_syn="((S/NP)\\NP)", \
                                                  target_shells=[
                                                      'lambda $0_{e}.lambda $1_{e}.lambda $2_{ev}.placeholderP($0,$1,$2)', \
                                                      'lambda $0_{e}.lambda $1_{e}.lambda $2_{ev}.placeholderP($1,$0,$2)'])
    elif options.phenom_type == 'trans_dax':
        phenom = extract_from_lexicon.get_dax_phenom()
        word_lfs = [('daxed', 'lambda $0_{e}.lambda $1_{e}.lambda $2_{ev}.v|dax&PAST($1,$0,$2)'), \
                    ('dax', 'lambda $0_{e}.lambda $1_{e}.lambda $2_{ev}.v|dax($1,$0,$2)')]
        phenom = extract_from_lexicon.Phenomenon('Daxed', word_lfs, '((S\\NP)/NP)', \
                                                  ['((S\\NP)/NP)', "((S/NP)/NP)", "((S\\NP)\\NP)", "((S/NP)\\NP)"], \
                                                  ['daxed', 'dax'])
        words = ['daxed', 'dax']
    elif options.phenom_type == 'det':
        word_lfs = extract_from_lexicon.get_det_lfs()
        phenom = extract_from_lexicon.Phenomenon('Determiners', word_lfs, '(NP/N)', \
                                                  ['(NP/N)', "(NP\\N)"], sem_type='(NP|N)')
    elif options.phenom_type == 'dax_det':
        word_lfs = [('jax', 'lambda $0_{<e,t>}.det|jax($1,$0($1))')]
        phenom = extract_from_lexicon.Phenomenon('Jax (deter.)', word_lfs, '(NP/N)', \
                                                  ['(NP/N)', "(NP\\N)"], sem_type='(NP|N)')
        words = ['jax']
    elif options.phenom_type == 'nouns':
        word_lfs = extract_from_lexicon.get_noun_lfs()
        phenom = extract_from_lexicon.Phenomenon('Nouns', word_lfs, 'N', ['N'], sem_type='N')
        # words = [x[0] for x in word_lfs]
    elif options.phenom_type == 'dax_nouns':
        word_lfs = [('zax', 'lambda $0_{e}.n|zax($0)')]
        phenom = extract_from_lexicon.Phenomenon('Zax Nouns', word_lfs, 'N', ["N"], sem_type='N')
        words = ['zax']
    elif options.phenom_type == 'prep':
        word_lfs = extract_from_lexicon.get_prep_lfs()
        phenom = extract_from_lexicon.Phenomenon('Prepositions', word_lfs, '(PP/NP)', \
                                                  ['(PP/NP)', "(PP\\NP)"], sem_type='(PP|NP)')
    elif options.phenom_type == 'prep_dax':
        word_lfs = [('ax', "lambda $0_{e}.lambda $1_{ev}.prep|ax($0,$1)")]
        phenom = extract_from_lexicon.Phenomenon('Ax Prepositions', word_lfs, '(PP/NP)', \
                                                  ['(PP/NP)', "(PP\\NP)"], sem_type='(PP|NP)')
        words = ['ax']
    else:
        sys.stderr.write('Error: phenom type not supported\n')
        sys.exit(1)

    results = {}
    for fns, suffix in zip(filenames, suffixes):
        pr_lf_given_w = draw_lf_given_w(phenom, words, fns, FIELD_NAMES_LF_GIVEN_W)
        pr_syn = draw_syn_cats(phenom.all_target_syns(), fns, FIELD_NAMES_SYN)
        pr_correct_syn = dict(pr_syn[phenom.target_syn()])
        for w, r in pr_lf_given_w.items():
            joint_syn_lf_prob = \
                [(sent_count, pr_lf_given_w * pr_correct_syn.get(sent_count, 0.0)) \
                 for sent_count, pr_lf_given_w in r]
            results[(suffix, w)] = joint_syn_lf_prob

    draw_graph(pr_syn, options)

    print(results)
    draw_graph(results, options)
