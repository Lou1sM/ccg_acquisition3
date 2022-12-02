from generator import generateSent
import re
from inside_outside_calc import i_o_oneChart
from parser import parse
from build_inside_outside_chart import build_chart
from sample_most_probable_parse import sample
from cat import Cat,SynCat
from make_graphs import output_cat_probs
import exp


def train_rules(lexicon,rule_set,sem_store, is_one_word, inputpairs, skip_q,
                cats_to_check, train_out, sentence_count=0,
                truncate_complex_exps=True, is_devel=False):

    datasize = 20000
    lexicon.set_learning_rates(datasize)
    train_limit = 20000
    max_sentence_length = 10
    line_count = 0
    sentence = None
    top_cat_list = []

    while line_count < len(inputpairs):
        line = inputpairs[line_count]
        line_count += 1
        if line[:5] == "Sent:":
            is_q = False
            sentence = line[6:].strip().rstrip()
            if sentence.count(" ") > max_sentence_length:
                print("rejecting: example too long ", line)
                sentence = None
                continue
            top_cat_list = []

        if sentence and line[:4] =="Sem:":
            semstring = line[5:].strip().rstrip()
            try:
                sem, _ = exp.make_exp_with_args(semstring, {})
                if not sem.to_string() == re.sub(r'_\d','',semstring):
                    breakpoint()
                    print('sem string doesn\'t match original')
            except (AttributeError, IndexError):
                print("LF could not be parsed\nSent : " + sentence)
                print("Sem: " + semstring + "\n\n")
                continue
            if len(sem.all_extractable_sub_exps()) > 9 and truncate_complex_exps:
                sentence = None
                continue

            try:
                is_q, sc = get_top_cat(sem)
            except IndexError:
                # print "couldn't determine syntactic category ", sem_line
                words = None
                sentence = None
                sem = None
                sc = None
                continue

            words = sentence.split()
            if not is_q and words[-1] in ["?", "."]:
                words = words[:-1]
            if len(words) == 0:
                words = None
                sentence = None
                sem = None
                sc = None
                continue

            top_cat = Cat(sc, sem)
            top_cat_list.append(top_cat)

        if sentence and line[:11] == "example_end":
            print("Sent : " + sentence, file=train_out)
            print("update weight = ", lexicon.get_learning_rate(sentence_count), file=train_out)
            print(sentence_count, file=train_out)
            for top_cat in top_cat_list:
                print("Cat : " + top_cat.to_string(), file=train_out)
            cat_store = {}
            if len(words) > 8 or (skip_q and "?" in sentence):
                sentence = []
                sem = None
                continue

            try:
                chart = build_chart(top_cat_list, words, rule_set, lexicon, cat_store, sem_store, is_one_word)
            except (AttributeError, IndexError):
                print("Sent : " + sentence)
                continue
            i_o_oneChart(chart, sem_store, lexicon, rule_set, True, 0.0, sentence_count)
            sentence_count += 1

            if is_devel or (sentence_count == train_limit):
                break
    return lexicon, rule_set, sem_store, chart

def get_top_cat(sem):
    if sem.check_if_wh():
        is_q = False
        sc = SynCat.swh
    elif sem.is_q():
        is_q = True
        sc = SynCat.q
    else:
        is_q = False
        sc = SynCat.all_syn_cats(sem.type())[0]
    return is_q, sc

def print_top_parse(chart, rule_set, output_fpath):
    topparses = []
    for entry in chart[len(chart)]:
        top = chart[len(chart)][entry]
        topparses.append((top.inside_score, top))
    top_parse = sample(sorted(topparses)[-1][1], chart, rule_set)

    with open(output_fpath, "w") as f:
        f.write('top parse:')
        print(top_parse,file=f)
        print(top.inside_score,file=f)

def print_cat_probs(cats_to_check, lexicon, sem_store, rule_set):
    print("outputting cat probs")
    # this samples the probabilities of each of the syn cat for a given type
    for c in cats_to_check:
        posType = c[0]
        lf_type = c[2]
        arity = c[3]
        # these go in cats
        cats = c[4]
        out_file = c[1]
        output_cat_probs(posType, lf_type, arity, cats, lexicon, sem_store, rule_set, out_file)

def generate_sentences(sentstogen, lexicon, rule_set, cat_store, sem_store, is_one_word, genoutfile, sentence_count):
    sentnum = 1
    for (gensent, gensemstr) in sentstogen:
        gensem = exp.make_exp_with_args(gensemstr, {})[0]
        if gensem.check_if_wh():
            sc = SynCat.swh
        elif gensem.is_q():
            sc = SynCat.q
        else:
            sc = SynCat.allSynCats(gensem.type())[0]
        genCat = Cat(sc, gensem)
        print("gonna generate sentence ", gensent)
        generateSent(lexicon, rule_set, genCat, cat_store, sem_store, is_one_word, gensent, genoutfile,
                     sentence_count, sentnum)
        sentnum += 1

def test(test_in, test_out, errors_out, sem_store, rule_set, current_lexicon, sentence_count):
    current_lexicon.refresh_all_params(sentence_count)
    retsem = None
    for line in test_in:
        if line[:5] == "Sent:":
            sentence = line[6:].split()
        if line[:4] == "Sem:":
            try:
                sem = exp.make_exp_with_args(line[5:].strip().rstrip(), {})[0]
            except IndexError:
                print(sentence, file=errors_out)
                print(line, file=errors_out)
                continue
            if not sem.is_q() and sentence[-1] in [".", "?"]:
                sentence = sentence[:-1]
            if len(sentence) == 0:
                sem = None
                sentence = None
                continue
            print(sentence, file=test_out)
            retsem = None
            top_parse = None
            #try:
            (retsem, top_parse, topcat) = parse(sentence, sem_store, rule_set, current_lexicon, sentence_count, test_out)
            #except (AttributeError, IndexError) as e:
                #print(e)
                #pass
            if retsem and sem and retsem.equals(sem):
                print(f"CORRECT\n{retsem.to_string(True)}\n{topcat.to_string()}", file=test_out)
            elif not retsem:
                print("\nNO PARSE\n\n", file=test_out)
                continue
            else:
                print(f"WRONG\n{retsem.to_string(True)}\n{topcat.to_string()}", file=test_out)
                print(sem.to_string(True), file=test_out)
                if sem and retsem.equals_placeholder(sem):
                    print(f"CORRECTPlaceholder\n{retsem.to_string(True)}\n{topcat.to_string()}", file=test_out)

            print(f'top parse:\n{top_parse}\n\n', file=test_out)
