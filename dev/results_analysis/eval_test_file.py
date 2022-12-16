import matplotlib

matplotlib.use('Agg')
import numpy as np
import sys, pylab
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 18})
STYLES = ['-', '--', ':', '-.', '-', '-']


def eval_file(filename):
    D = {}
    f = open(filename)
    for line in f:
        line = line.strip()
        if line in ['CORRECT', 'CORRECTPlaceholder', 'NO PARSE', 'WRONG']:
            D[line] = D.get(line, 0) + 1
        if line == 'CORRECTPlaceholder':
            D['WRONG'] = D.get('WRONG', 0) - 1
    tot = sum(D.values())
    print(D)
    print('Number of instances: ' + str(tot))

    if tot > 0:
        acc = 1.0 * D.get('CORRECT', 0) / tot
        acc_placeholder = 1.0 * (D.get('CORRECT', 0) + \
                                 D.get('CORRECTPlaceholder', 0)) / tot
        return acc, acc_placeholder
    else:
        return None, None


def get_results(filenames):
    """returns the results array for a single set of files"""
    results = [(0, 0, 0)]
    for ind, fn in enumerate(filenames.split(':')):
        acc, acc_placeholder = eval_file(fn)
        if acc is not None:
            results.append((ind + 1, acc, acc_placeholder))
    return np.array(results)


if len(sys.argv) < 3:
    print('Usage: eval_test_file.py <output graph> ' + \
          '<test files (: delimited)> <test files (: delimited)> ...')
    sys.exit(-1)

results = []
for filenames in sys.argv[2:]:
    if filenames[-1] == ':':
        filenames = filenames[:-1]
    results.append(get_results(filenames))

for ind, res in enumerate(results):
    plt.plot(res[:, 0], res[:, 1], STYLES[ind], linewidth=3.0, alpha=1 - ind * 0.2)

plt.xlabel('Section #')
plt.ylabel('Accuracy')
# plt.suptitle('LF Prediction Accuracy (no guessing)')
plt.legend(['No distractors', '2 distractors', '4 distractors', '6 distractors'], 'best', prop={'size': 12})
plt.ylim([0.0, 1.0])
plt.savefig(sys.argv[1] + '_no_guessing.png')

plt.clf()

for ind, res in enumerate(results):
    plt.plot(res[:, 0], res[:, 2], STYLES[ind], linewidth=3.0)

plt.xlabel('Section #')
plt.ylabel('Accuracy')
# plt.suptitle('LF Prediction Accuracy (with guessing)')
plt.ylim([0.0, 1.0])
plt.legend(['No distractors', '2 distractors', '4 distractors', '6 distractors'], 'best', prop={'size': 12})
plt.savefig(sys.argv[1] + '_with_guessing.png')
