import sys, pylab, pdb, os
import numpy as np
import matplotlib.pyplot as plt
import extract_from_lexicon3
from optparse import OptionParser

FIELD_NAMES = ['name', 'sentence count', 'row type', 'LF', 'word', 'prob']


def draw_syn_given_lf_graph(phenom, words, file_list, field_names):
    results = dict([(w, []) for w in words])
    seen_sentence_counts = set([0])
    for fn in file_list:
        f = open(fn)
        for line in f:
            line = line.strip()
            if "Pr(correct syn|LF)" in line:
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


def draw_graph(results):
    legend = []
    for w, res in results.items():
        arr = np.array(res)
        arr = arr[arr[:, 0].argsort(),]
        plt.plot(arr[:, 0], arr[:, 1], linewidth=3.0)
        legend.append(w)

    plt.legend(legend, loc='lower right')
    plt.show()


def cmd_line_parser():
    """
    Returns the command line parser.
    """
    usage = "usage: %prog [options]\n"
    opt_parser = OptionParser(usage=usage)
    opt_parser.add_option("-d", action="store", dest="direc",
                          help="the target directory")
    opt_parser.add_option("-s", action="store", dest="suffix", default='',
                          help="suffix")
    opt_parser.add_option("-l", action="store", dest="filelist", default='',
                          help="the list of files to use (: delimited)")
    # opt_parser.add_option("-p", action="store", dest='phenom_string',
    #                      help="the phenomenon to observe")
    opt_parser.add_option("-w", action="store", dest="words",
                          help="the list of words to observe (: delimited)")

    return opt_parser


if __name__ == '__main__':
    parser = cmd_line_parser()
    options, args = parser.parse_args(sys.argv)

    filelist = options.filelist.split(':')
    if options.direc:
        filenames = [fn for fn in os.listdir(options.direc) if fn.endswith(options.suffix) and fn in filelist]
    else:
        filenames = [fn for fn in filelist if fn.endswith(options.suffix)]
    print(filenames)

    words = options.words.split(':')

    word_lfs = extract_from_lexicon3.get_transitive_lfs()
    phenom = extract_from_lexicon3.Phenomenon('Transitives', word_lfs, '((S\\NP)/NP)', \
                                              ["((S\\NP)/NP)", "((S/NP)/NP)", "((S\\NP)\\NP)", "((S/NP)\\NP)"], \
                                              two_LFs=True, \
                                              flipped_target_syn="((S/NP)\\NP)", \
                                              target_shells=[
                                                  'lambda $0_{e}.lambda $1_{e}.lambda $2_{ev}.placeholderP($0,$1,$2)', \
                                                  'lambda $0_{e}.lambda $1_{e}.lambda $2_{ev}.placeholderP($1,$0,$2)'])

    results = draw_syn_given_lf_graph(phenom, words, filenames, FIELD_NAMES)
    draw_graph(results)
