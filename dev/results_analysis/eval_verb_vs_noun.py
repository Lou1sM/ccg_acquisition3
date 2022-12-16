import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18})

import sys
import matplotlib.pyplot as plt
import numpy as np
import find_new_nouns
import pdb

FIELD_NAMES = ['name', 'sentence count', 'row type', 'LF', 'word', 'prob']


def check_monotonic(L):
    """
    Receives a list of sets. Checks if each set contains the former. Otherwise aborts
    """
    last_S = set()
    for S in L:
        if not last_S.issubset(S):
            print(S, last_S)
            sys.exit(-1)
        last_S = S


def get_curve(thresh, output_file, inp_filenames, category, ratio):
    learned_ratio = []
    sentence_counts = []
    all_words = []
    for fn in [x for x in inp_filenames.split(':') if x != '']:
        not_learned_list = []
        learned_list = []
        cur_sent_count = None
        f = open(fn)
        for line in f:
            if ("Pr(correct word|LF)" in line) and (category in line):
                fields = dict(zip(FIELD_NAMES, line.split('\t')))
                cur_sent_count = int(fields['sentence count'])
                if float(fields['prob']) >= thresh:
                    learned_list.append(fields['word'])
                else:
                    not_learned_list.append(fields['word'])
        all_words.append((cur_sent_count, set(learned_list + not_learned_list)))
        if ratio:
            learned_ratio.append(1.0 * len(learned_list) / (len(not_learned_list) + len(learned_list)))
        else:
            learned_ratio.append(len(learned_list))
        sentence_counts.append(cur_sent_count)
    all_words.sort(key=lambda x: x[0])
    check_monotonic([x[1] for x in all_words])
    learned_ratio = [0] + learned_ratio
    sentence_counts = [0] + sentence_counts
    return learned_ratio, sentence_counts


def main(args):
    if len(args) != 5:
        print(
            'Usage eval_verb_vs_noun.py <files : delimited> <thresh> <output file> <T to display ratio, abs size otherwise>')
        sys.exit(-1)
    thresh = float(args[2])
    output_file = args[3]
    ratio = (args[4] == 'T')

    ratio_nouns, sc_nouns = get_curve(thresh, output_file, args[1], 'Nouns', ratio)
    ratio_verbs, sc_verbs = get_curve(thresh, output_file, args[1], 'Transitives', ratio)

    # plt.xlim([0,1500])
    if ratio:
        plt.ylim([0.0, 1.0])
    plt.plot(np.array(sc_nouns), np.array(ratio_nouns), 'o-', linewidth=3.0)
    plt.plot(np.array(sc_verbs), np.array(ratio_verbs), 'o-', linewidth=3.0, alpha=0.8)
    plt.legend(['Nouns', 'Transitive Verbs'], 'best', prop={'size': 12})
    plt.xlabel('#Seen Utterances')
    plt.ylabel('#Learned Words')
    # plt.suptitle('Production Vocabulary Size for Nouns and Transitives')


    plt.savefig(output_file)


if __name__ == '__main__':
    main(sys.argv)
