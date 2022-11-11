import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18})

import sys, pylab, pdb, os, re
import numpy as np
import matplotlib.pyplot as plt
import extract_from_lexicon3
from optparse import OptionParser

COLORS = ['b', 'g', 'r', 'c', 'm', 'y']
STYLES = ['-', ':', '--', '-.', '-', '-', '-', '-']
ALL_SUFFIXES = 'ALL_SUFFIXES'
FIELD_NAMES_LF_GIVEN_W = ['name', 'sentence count', 'row type', 'LF', 'word', 'prob']
FIELD_NAMES_W_GIVEN_LF = ['name', 'sentence count', 'row type', 'LF', 'word', 'prob']
FIELD_NAMES_SYN_GIVEN_LF = ['name', 'sentence count', 'row type', 'LF', 'word', 'prob']
FIELD_NAMES_SYN = ['name', 'sentence count', 'syn_cat', 'row_type', 'prob']

REGARD_TWO_LFS = True


def draw_w_given_lf(phenom, words, filenames, field_names, row_type):
    results = dict([(w, []) for w in words])
    seen_sentence_counts = set([0])
    for fn in filenames:
        f = open(fn)
        for line in f:
            line = line.strip()
            if (row_type in line):
                fields = dict(zip(field_names, line.split('\t')))
                seen_sentence_counts.update([int(fields['sentence count'])])
                if fields['word'] in words and \
                                phenom.target_lf(fields['word']) == fields['LF']:
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
        res.append((0, 1.0 / len(phenom.all_target_syns())))
        results[syncat] = res
    return results


def get_const_occ_marker(w, res, const_occ, phenom):
    word = w[1]
    p = re.compile("[\,\.\(\)][^\,\.\(\)\|]+\|[^\,\.\(\)\|]+[\,\.\(\)]")
    lf_const = p.findall(phenom.target_lf(word))[0][1:-1]
    res_x = [x[0] for x in res]
    return [min(z for z in res_x if z >= max(y, 1)) for y in const_occ[lf_const]]


def draw_graph(results, options, show_plot=True, save_plot=None, suffixes=None, \
               ylabel=None, title_text=None, xlim=None, \
               const_occ=None, phenom=None, draw_markers=False):
    def _convert_to_legend(s):
        if s == "((S\NP)/NP)":
            return 'Subject-Verb-Object'
        elif s == "((S/NP)/NP)":
            return 'Verb initial'
        elif s == "((S\NP)\NP)":
            return "Verb final"
        elif s == "((S/NP)\NP)":
            return 'Object-Verb-Subject'
        elif s == "(NP/N)":
            return "Pre-nominal"
        elif s == "(NP\N)":
            return "Post-nominal"
        elif s == "(PP/NP)":
            return "Preposition"
        elif s == "(PP\NP)":
            return "Postposition"
        else:
            return s

    plt.clf()
    # legend = []
    ind = 0

    for w, res in results.items():
        arr = np.array(res)
        arr = arr[arr[:, 0].argsort(),]
        arr[arr == 0] = 0.01
        arr[arr == 1] = 0.99

        plt.xlabel('#Seen Utterances')

        if ylabel:
            plt.ylabel(ylabel)
        else:
            plt.ylabel('???')

        if title_text:
            plt.suptitle(title_text)

        if suffixes and len(suffixes) == 1:
            label = w[1]  # legend.append(w[1])
        elif w[0] == 'out4':
            label = w[1] + ' low freq. arg.'  # legend.append(w[1]+' low freq. arg.')
        elif w[0] == 'out8':
            label = w[1] + ' high freq. arg.'  # legend.append(w[1]+' high freq. arg.')
        elif w[0] == 'out10':
            label = w[1] + ' modal'  # legend.append(w[1]+' modal')
        elif w[0] == 'out11':
            label = w[1] + ' NP+PP'
            # legend.append(w[1]+' NP+PP')
        elif w[0] == 'out5':
            label = 'dax example #1'  # legend.append('dax example #1')
        elif w[0] == 'out6':
            label = 'dax example #2'
            # legend.append('dax example #2')
        else:
            label = _convert_to_legend(w)  # legend.append(_convert_to_legend(w))

        if draw_markers:
            const_occ_ticks = get_const_occ_marker(w, res, const_occ, phenom)
            v = -0.05 * (1 + ind) * np.ones(len(const_occ_ticks) + 2)
            plt.plot([-1000] + const_occ_ticks + [5000], v, COLORS[ind], marker='|', linewidth=3.0, alpha=1 - ind * 0.1,
                     mew=5, ms=10)

        curve = plt.plot(arr[:, 0], arr[:, 1], STYLES[ind] + COLORS[ind], linewidth=3.0, alpha=1 - ind * 0.1,
                         label=label)
        ind += 1

    if draw_markers:
        plt.ylim([-0.3, 1.0])
        plt.yticks(np.arange(0, 1.0, 0.2))
    else:
        plt.ylim([0.0, 1.0])
    plt.xlim(0, xlim)

    if not options.nolegend:
        plt.legend(loc='best', prop={'size': 12})
        #    plt.legend(legendEntries,legendText,loc='best',prop={'size':12})
    if save_plot:
        plt.savefig(save_plot)
    if show_plot:
        plt.show()


def get_all_phenoms():
    all_phenoms = {}

    word_lfs = extract_from_lexicon3.get_transitive_lfs()
    phenom = extract_from_lexicon3.Phenomenon('Transitives', word_lfs, '((S\NP)/NP)', \
                                              ["((S\NP)/NP)", "((S/NP)/NP)", "((S\NP)\NP)", "((S/NP)\NP)"], \
                                              two_LFs=True, \
                                              flipped_target_syn="((S/NP)\NP)", \
                                              target_shells=[
                                                  'lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.placeholderP($0,$1,$2)', \
                                                  'lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.placeholderP($1,$0,$2)'])
    all_phenoms['trans'] = phenom

    phenom = extract_from_lexicon3.get_dax_phenom()
    word_lfs = [('daxed', 'lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.v|dax&PAST($1,$0,$2)'), \
                ('dax', 'lambda $0_{e}.lambda $1_{e}.lambda $2_{r}.v|dax($1,$0,$2)')]
    phenom = extract_from_lexicon3.Phenomenon('Nonce Transitive', word_lfs, '((S\NP)/NP)', \
                                              ['((S\NP)/NP)', "((S/NP)/NP)", "((S\NP)\NP)", "((S/NP)\NP)"], \
                                              ['daxed', 'dax'], two_LFs=True)
    all_phenoms['trans_dax'] = phenom

    word_lfs = extract_from_lexicon3.get_det_lfs()
    phenom = extract_from_lexicon3.Phenomenon('Determiners', word_lfs, '(NP/N)', \
                                              ['(NP/N)', "(NP\N)"], sem_type='(NP|N)')
    all_phenoms['det'] = phenom

    word_lfs = [('jax', 'lambda $0_{<e,t>}.det|jax($1,$0($1))')]
    phenom = extract_from_lexicon3.Phenomenon('Jax (deter.)', word_lfs, '(NP/N)', \
                                              ['(NP/N)', "(NP\N)"], sem_type='(NP|N)')
    all_phenoms['dax_det'] = phenom

    word_lfs = extract_from_lexicon3.get_noun_lfs()
    phenom = extract_from_lexicon3.Phenomenon('Nouns', word_lfs, 'N', ['N'], sem_type='N')
    all_phenoms['nouns'] = phenom

    word_lfs = [('zax', 'lambda $0_{e}.n|zax($0)')]
    phenom = extract_from_lexicon3.Phenomenon('Nonce Noun', word_lfs, 'N', ["N"], sem_type='N')
    all_phenoms['dax_nouns'] = phenom

    word_lfs = extract_from_lexicon3.get_prep_lfs()
    phenom = extract_from_lexicon3.Phenomenon('Prepositions', word_lfs, '(PP/NP)', \
                                              ['(PP/NP)', "(PP\NP)"], sem_type='(PP|NP)')
    all_phenoms['prep'] = phenom

    word_lfs = [('ax', "lambda $0_{e}.lambda $1_{r}.prep|ax($0,$1)")]
    phenom = extract_from_lexicon3.Phenomenon('Nonce Preposition', word_lfs, '(PP/NP)', \
                                              ['(PP/NP)', "(PP\NP)"], sem_type='(PP|NP)')
    all_phenoms['prep_dOBax'] = phenom

    word_lfs = extract_from_lexicon3.get_intransitive_lfs()
    phenom = extract_from_lexicon3.Phenomenon('Intransitives', word_lfs, '(S\NP)', \
                                              ['(S\NP)', "(S/NP)"], sem_type='(S|NP)')
    all_phenoms['intrans'] = phenom

    word_lfs = [('corp', "lambda $0_{e}.lambda $1_{r}.prep|corp($0,$1)")]
    phenom = extract_from_lexicon3.Phenomenon('Nonce Preposition', word_lfs, '(PP/NP)', \
                                              ['(PP/NP)', "(PP\NP)"], sem_type='(PP|NP)')
    all_phenoms['corp_prep'] = phenom

    word_lfs = [('corp', 'lambda $0_{e}.n|corp($0)')]
    phenom = extract_from_lexicon3.Phenomenon('Nonce Noun', word_lfs, 'N', \
                                              ["N"], sem_type='N')
    all_phenoms['corp_noun'] = phenom

    return all_phenoms


def add_curve(results, fns, suffix, phenom, words):
    """appends results with a new curve"""
    pr_syn = draw_syn_cats(phenom.all_target_syns(), fns, FIELD_NAMES_SYN)
    if phenom.two_LFs():
        pr_w_given_lf = draw_w_given_lf(phenom, words, fns, FIELD_NAMES_W_GIVEN_LF,
                                        row_type="Pr(correct word|LF1 or LF2)")
        pr_syn_given_lf = draw_w_given_lf(phenom, words, fns, FIELD_NAMES_SYN_GIVEN_LF,
                                          row_type="Pr(correct syn|LF1 or LF2)")
        for w in pr_w_given_lf.keys():
            r_w_given_lf = dict(pr_w_given_lf[w])
            r_syn_given_lf = dict(pr_syn_given_lf[w])
            joint_syn_w_prob = \
                [(sent_count, r_w_given_lf[sent_count] * r_syn_given_lf.get(sent_count, 0.0)) \
                 for sent_count in r_w_given_lf.keys()]
            results[(suffix, w)] = joint_syn_w_prob
    else:
        pr_w_given_lf = draw_w_given_lf(phenom, words, fns, FIELD_NAMES_W_GIVEN_LF, row_type="Pr(correct word|LF)")
        pr_correct_syn = dict(pr_syn[phenom.target_syn()])
        for w, r in pr_w_given_lf.items():
            joint_syn_w_prob = \
                [(sent_count, pr_w_given_lf * pr_correct_syn.get(sent_count, 0.0)) \
                 for sent_count, pr_w_given_lf in r]
            results[(suffix, w)] = joint_syn_w_prob
    return results, pr_syn


def cmd_line_parser():
    """
    Returns the command line parser.
    """
    usage = "usage: %prog [options]\n"
    opt_parser = OptionParser(usage=usage)
    opt_parser.add_option("-d", action="store", dest="direc",
                          help="directory to take files from")
    opt_parser.add_option("-s", action="store", dest="suffixes",
                          help="the suffixes for the files")
    opt_parser.add_option("-c", action="store", dest="infix", default='',
                          help="select only files that contain this as a sub-string")
    opt_parser.add_option("-p", action="store", dest="phenom_type",
                          help="the phenomenon to inspect")
    opt_parser.add_option("--nolegend", action="store_true", dest="nolegend",
                          help="omit the legend")
    opt_parser.add_option("-o", action="store", dest="output_graph",
                          help="the file to output the graphs to")
    opt_parser.add_option("--draw_syn", action="store", dest="draw_syn",
                          help="whether to draw the syntactic generalization or not")
    opt_parser.add_option("--xlim", action="store", type="float", dest="xlim",
                          help="the limit of the x-axis")
    opt_parser.add_option("--add", action="store", dest="additional_words",
                          help="additional words to add :-delimited. e.g., duck,nouns:eat,trans")
    opt_parser.add_option("--occ", action="store_true", dest="occ_lines",
                          help="adding the lines which show the occurances of the indiv words")

    return opt_parser


####################
# MAIN
####################

if __name__ == '__main__':
    parser = cmd_line_parser()
    options, args = parser.parse_args(sys.argv)

    direc = options.direc
    if options.suffixes:
        suffixes = options.suffixes.split(':')
        filenames = []
        for suffix in suffixes:
            filenames.append(
                [direc + '/' + fn for fn in os.listdir(direc) if fn.endswith(suffix) and options.infix in fn])
    else:
        suffixes = [ALL_SUFFIXES]
        filenames = [[direc + '/' + fn for fn in os.listdir(direc) if options.infix in fn]]

    all_phenoms = get_all_phenoms()
    if options.phenom_type in all_phenoms.keys():
        phenom = all_phenoms[options.phenom_type]
    else:
        sys.stderr.write('Error: phenom type not supported\n')
        sys.exit(1)

#    const_occ_f = open("allConsts")
#    const_occ = {}
#    for line in const_occ_f:
#        fields = line.strip().split()
#        L = const_occ.get(fields[1], [])
#        L.append(int(fields[0]))
#        const_occ[fields[1]] = L

    if options.phenom_type == 'trans':
        words = ['want', 'see', 'put', 'have', 'spank', 'hugged', 'tied', 'reach']  # ['put','see','break','leave']
        draw_markers = True
    elif options.phenom_type == 'trans_dax':
        words = ['daxed', 'dax']
        draw_markers = False
    elif options.phenom_type == 'det':
        words = ['his', 'any', 'a', 'more', 'the']
        draw_markers = True
    elif options.phenom_type == 'dax_det':
        words = ['jax']
        draw_markers = False
    elif options.phenom_type == 'nouns':
        words = ['boy', 'water', 'truck', 'pencil', 'time', 'way', 'car', 'name', 'crane', 'rabbit', 'tank', 'puppy', 'barber', 'raccoon', 'film', 'soap']  # ['pencil', 'cookie', 'girl', 'shoe', 'balloon']
        draw_markers = True
    elif options.phenom_type == 'dax_nouns':
        words = ['zax']
        draw_markers = False
    elif options.phenom_type == 'prep':
        words = ['with', 'at', 'in', 'about', 'on']
        draw_markers = True
    elif options.phenom_type == 'prep_dax':
        words = ['ax']
        draw_markers = False
    elif options.phenom_type == 'corp_noun':
        words = ['corp']
        draw_markers = False
    elif options.phenom_type == 'corp_prep':
        words = ['corp']
        draw_markers = False

    if not options.occ_lines:
        draw_markers = False

    print(phenom.name())
    results = {}
    for fns, suffix in zip(filenames, suffixes):
        results, pr_syn = add_curve(results, fns, suffix, phenom, words)

    if not options.additional_words and not options.draw_syn:
        draw_graph(pr_syn, options, save_plot=options.output_graph + '_syn.png', show_plot=False,
                   ylabel='Relative Prob.', draw_markers=False)
    # ,title_text='Learned Prior for '+phenom.name())

    if options.additional_words:
        word_phenom_dir_suffix_tuples = [x.split(',') for x in options.additional_words.split(':')]
        for word, target_phenom, cur_direc, cur_suffix in word_phenom_dir_suffix_tuples:
            fns = [cur_direc + '/' + fn for fn in os.listdir(cur_direc) if
                   fn.endswith(cur_suffix) and options.infix in fn]
            results, temp = add_curve(results, fns, cur_suffix, all_phenoms[target_phenom], [word])

    if options.draw_syn:
        phenom_dir_suffix_tuples = [x.split(',') for x in options.draw_syn.split(':')]
        for syn_phenom, cur_direc, cur_suffix in phenom_dir_suffix_tuples:
            fns = [cur_direc + '/' + fn for fn in os.listdir(cur_direc) if
                   fn.endswith(cur_suffix) and options.infix in fn]
            pr_syn = draw_syn_cats(all_phenoms[syn_phenom].all_target_syns(), fns, FIELD_NAMES_SYN)
            pr_correct_syn = dict(pr_syn[all_phenoms[syn_phenom].target_syn()])
            results[(suffix, syn_phenom)] = list(pr_correct_syn.items())

#    draw_graph(results, options, save_plot=options.output_graph + '_individual.png', show_plot=False, \
#               suffixes=suffixes, ylabel='CPP', xlim=options.xlim, phenom=phenom, \
#               const_occ=const_occ, draw_markers=draw_markers)  # ,title_text='CPP for '+phenom.name(),)

    draw_graph(results, options, save_plot=options.output_graph + '_individual.png', show_plot=False, \
               suffixes=suffixes, ylabel='CPP', xlim=options.xlim, phenom=phenom, \
               draw_markers=draw_markers)  # ,title_text='CPP for '+phenom.name(),)
