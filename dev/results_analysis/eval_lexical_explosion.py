import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18})

import sys
import matplotlib.pyplot as plt
import numpy as np
import find_new_nouns
import pdb

FIELD_NAMES = ['name', 'sentence count', 'row type', 'LF', 'word', 'prob']


def main(args):
    if len(args) != 4:
        print('Usage eval_lexical_explosion.py <files : delimited> <thresh> <output file>')
        sys.exit(-1)
    thresh = float(args[2])
    output_file = args[3]
    learned_ratio = []
    sentence_counts = []
    for fn in [x for x in args[1].split(':') if x != '']:
        not_learned_list = []
        learned_list = []
        non_zero_list = []
        cur_sent_count = None
        f = open(fn)
        for line in f:
            if ("Pr(correct word|LF)" in line) and ("Nouns" in line):
                fields = dict(zip(FIELD_NAMES, line.split('\t')))
                cur_sent_count = int(fields['sentence count'])
                if float(fields['prob']) >= thresh:
                    learned_list.append(fields['word'])
                elif float(fields['prob']) > 0:
                    non_zero_list.append(fields['word'])
                else:
                    not_learned_list.append(fields['word'])
        learned_ratio.append(1.0 * len(learned_list))  # / (len(not_learned_list) + len(learned_list)))
        sentence_counts.append(cur_sent_count)
        print(cur_sent_count)
        print(' '.join(learned_list))

    print(learned_ratio)
    learned_ratio = [0] + learned_ratio
    sentence_counts = [0] + sentence_counts
    plt.xlim([0, 600])
    plt.ylim([0, 55])
    plt.ylabel('#Learned Nouns')
    plt.xlabel('#Seen Utterances')
    # plt.suptitle('Early Production Vocabulary Size for Nouns')

    plt.plot(np.array(sentence_counts), np.array(learned_ratio), 'o-', linewidth=3.0)
    plt.savefig(output_file)
    # plt.show()


    # total_nouns = find_new_nouns.main()
    # x = [e[0] for e in total_nouns]
    # y = [e[1] for e in total_nouns]
    # total_nouns_dict = dict(total_nouns)
    # plt.plot(np.array(x),np.array(y),linewidth=3.0)
    # plt.legend(['learned nouns', 'total nouns'])


    """
    plt.plot(np.array(sentence_counts),\
                 np.array([1.0*learned_count/total_nouns_dict[sc] \
                               for sc,learned_count in zip(sentence_counts,learned_ratio)]),\
                 linewidth=3.0)
    #plt.show()
    """


if __name__ == '__main__':
    main(sys.argv)
