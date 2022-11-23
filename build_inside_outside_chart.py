# Need to build inside-outside charts
# slight difference from usual because, we don't have
# one to one word, sentence mappings. but that's ok.
# Hold as dictionary?
#     (head,p,q)

from tools import inf
import exp


# purpose of SemStore is to save 'all the sem reps'
class SemStore:
    def __init__(self):
        self.store = {}

    def clear(self):
        self.store = {}

    def check(self, sem_key):
        if not isinstance(sem_key, str):
            raise Exception

        if sem_key not in self.store:
            return False
        else:
            return True

    def get_log_prior(self, sem_key):
        if not self.check(sem_key):
            sem = exp.make_exp_with_args(sem_key, {})
            # OMRI ADDED THE NEXT TWO LINES
            if isinstance(sem, tuple):
                sem = sem[0]
        else:
            sem = self.store[sem_key]
        return sem.semprior()

    def add(self, sem):
        self.store[sem.to_string(True)] = sem
        if sem.to_string_shell(True) not in self.store:
            self.store[sem.to_string_shell(True)] = sem.make_shell({})

    def get(self, sem_key):
        if sem_key in self.store:
            return self.store[sem_key]
        else:
            return None

        # Each node for each span must be realisable as a word
        # class word_target:
        # def __init__(self,word):
        # self.word = word
        # self.inside_prob = 0.0
        # self.outside_prob = 0.0

# chart entries are for syntactic nodes that represent the span
# p->q of the sentence
class ChartEntry:
    def __init__(self, ccgCat, p, q, sentence):
        self.sentence = sentence
        if p is not None and q is not None:
            self.words = sentence[p:q]
        else:
            self.words = sentence
        self.ccgCat = ccgCat
        self.syn_key = ccgCat.syn_string()
        self.sem_key = ccgCat.sem_string()
        self.p = p
        self.q = q
        self.word_prob = 0.0
        self.word_score = -inf
        self.sem_prob = 0.0
        self.sem_score = -inf
        self.inside_prob = 0.0
        self.inside_score = -inf
        self.max_score = -inf
        self.outside_prob = 0.0
        self.outside_score = -inf
        self.children = []
        if p is not None and q is not None:
            self.word_target = ' '.join(sentence[p:q])
        else:
            self.word_target = ' '.join(sentence)
        self.parents = []
        self.numParses = 0

    def lexKey(self):
        return self.word_target + " :: " + self.syn_key + " :: " + self.sem_key

    def to_string(self):
        return str(self.p) + ":" + str(
            self.q) + " :: " + self.word_target + " :: " + self.syn_key + " :: " + self.sem_key

    def add_parent(self, parent, side):
        instance = len(parent.children)
        self.parents.append((parent, instance, side))

    def addNumParses(self, np):
        self.numParses += np

    def getNumParses(self):
        return self.numParses

    def add_child(self, child):
        self.children.append(child)

    def clear_probs(self):
        self.word_prob = 0.0
        self.sem_prob = 0.0
        self.inside_prob = 0.0
        self.outside_prob = 0.0

    def get_inside(self):
        return self.inside_prob


def expand_chart(entry, chart, cat_store, sem_store, rule_set, lexicon, is_exclude_mwe, correct_index):
    """CatStore is a dictionary that maps pairs of syntactic and semantic forms
    to the set of pairs they can decompose to. It's a cache essentially.
    """
    if entry.ccgCat.sem.get_is_null(): return
    if entry.p < entry.q - 1:

        for d in range(entry.p + 1, entry.q):
            words_l = ' '.join(entry.sentence[entry.p:d])
            words_r = ' '.join(entry.sentence[d:entry.q])

            for pair in entry.ccgCat.all_pairs(cat_store):
                l_cat = pair[0]
                l_syncat = l_cat.syn
                l_syn = l_cat.syn_string()
                l_sem = l_cat.sem_string()

                r_cat = pair[1]
                r_syncat = r_cat.syn
                r_syn = r_cat.syn_string()
                r_sem = r_cat.sem_string()

                direction = pair[2]
                numcomp = pair[3]

                if not sem_store.check(l_sem): sem_store.add(l_cat.sem)
                if not sem_store.check(r_sem): sem_store.add(r_cat.sem)

                # - this is needed to build an actual parse - #
                rule_set.check_rule(entry.ccgCat.syn_string(), l_syncat, r_syncat, direction, numcomp)
                rule_set.check_rule(l_syn, None, None, None, None)
                rule_set.check_rule(r_syn, None, None, None, None)

                if correct_index:
                    lexicon.cur_cats.extend([r_cat, l_cat])

                lexicon.check(words_l, l_syn, l_sem, l_cat.sem)
                lexicon.check(words_r, r_syn, r_sem, r_cat.sem)

                # sem_store
                if (l_syn, l_sem, entry.p, d) not in chart[d - entry.p]:
                    cl = ChartEntry(l_cat, entry.p, d, entry.sentence)
                    chart[d - entry.p][(l_syn, l_sem, entry.p, d)] = cl
                    expand_chart(cl, chart, cat_store, sem_store, rule_set, lexicon, is_exclude_mwe, correct_index)
                chart[d - entry.p][(l_syn, l_sem, entry.p, d)].add_parent(entry, 'l')
                if (r_syn, r_sem, d, entry.q) not in chart[entry.q - d]:
                    cr = ChartEntry(r_cat, d, entry.q, entry.sentence)
                    chart[entry.q - d][(r_syn, r_sem, d, entry.q)] = cr
                    expand_chart(cr, chart, cat_store, sem_store, rule_set, lexicon, is_exclude_mwe, correct_index)
                chart[entry.q - d][(r_syn, r_sem, d, entry.q)].add_parent(entry, 'r')
                entry.add_child(((l_syn, l_sem, entry.p, d), (r_syn, r_sem, d, entry.q)))


def build_chart(top_cat_list, sentence, rule_set, lexicon, cat_store, sem_store, is_exclude_mwe):
    chart = {i:{} for i in range(1, len(sentence) + 1)}
    correct_index = (len(top_cat_list) - 1) / 2  # the index of the correct semantics

    for ind, top_cat in enumerate(top_cat_list):
        c1 = ChartEntry(top_cat, 0, len(sentence), sentence)
        if not sem_store.check(top_cat.sem.to_string(True)):
            sem_store.add(top_cat.sem)
        chart[len(sentence)][(top_cat.syn_string(), top_cat.sem_string(), 0, len(sentence))] = c1
        rule_set.check_start_rule(top_cat.syn)
        rule_set.check_rule(top_cat.syn_string(), None, None, None, None)
        wordspan = ' '.join(sentence)
        lexicon.check(wordspan, top_cat.syn_string(), top_cat.sem_string(), top_cat.sem)
        expand_chart(c1, chart, cat_store, sem_store, rule_set, lexicon, is_exclude_mwe, correct_index == ind)

    chart_size = 0
    for level in chart:
        chart_size += len(chart[level])

    return chart
