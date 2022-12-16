import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18})
import sys, pylab, pdb, os
import numpy as np
import matplotlib.pyplot as plt

FIELD_NAMES = ['sentence count', 'syn_cat', 'prob']
LEGEND = lambda x: x
STYLES = ['-', ':', '--', '-.', '-o', '-x', '-', '-']


def get_curves(filenames, cat_names=['SVO', 'OVS', 'SOV', 'OSV', 'VOS', 'VSO'], field_names=FIELD_NAMES):
    results = {}
    for sc in cat_names:
        results[sc] = [(0, 1.0 / 6)]

    seen_sentence_counts = set()
    for fn in filenames:
        f = open(fn)
        for line in f:
            line = line.strip()
            fields = dict(zip(field_names, line.split('\t')))
            seen_sentence_counts.update([int(fields['sentence count'])])
            if fields['syn_cat'] in cat_names:
                entry = results[fields['syn_cat']]
                entry.append((int(fields['sentence count']), float(fields['prob'])))
                results[fields['syn_cat']] = entry
    for syncat, res in results.items():
        cur_sentence_counts = set([int(r[0]) for r in res])
        for sc in seen_sentence_counts - cur_sentence_counts:
            res.append((sc, 0.0))
        results[syncat] = res
    return results


def draw_graph(results, save_plot=None, show_plot=False, title_text=False):
    legend = []
    # plt.xlim((0,2000))
    ind = 0
    for w, res in results.items():
        arr = np.array(res)
        arr = arr[arr[:, 0].argsort(),]
        arr[arr == 0] = 0.01
        arr[arr == 1] = 0.99
        plt.plot(arr[:, 0], arr[:, 1], STYLES[ind], linewidth=3.0, alpha=1 - ind * 0.1)
        legend.append(LEGEND(w))
        ind += 1
    plt.legend(legend, loc='best', prop={'size': 12})

    plt.xlabel('#Seen Utterances')
    plt.ylabel('Relative Prob.')
    if title_text:
        plt.suptitle(title_text)
    plt.ylim([0.0, 1.0])

    if save_plot:
        plt.savefig(save_plot)
    if show_plot:
        plt.show()


if len(sys.argv) not in [3, 4]:
    print('Usage: draw_syn_cats_graph.py <files :-delimited> <output graph> <title text>')
    sys.exit(-1)

fns = [x for x in sys.argv[1].split(':') if x != '']
syn_results = get_curves(fns)
if len(sys.argv) == 3:
    title_text = False
else:
    title_text = True
draw_graph(syn_results, sys.argv[2], title_text=False)
