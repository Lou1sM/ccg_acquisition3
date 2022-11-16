import pickle
from generator import generateSent
from inside_outside_calc import i_o_oneChart
from parser import parse
from build_inside_outside_chart import build_chart
from sample_most_probable_parse import sample
from cat import synCat
import cat
import exp


def train_rules(lexicon,RuleSet,sem_store, is_one_word, inputpairs, skip_q,
                cats_to_check, train_out, test_out, sentence_count=0,
                truncate_complex_exps=True, is_devel=False):

    datasize = 20000
    lexicon.set_learning_rates(datasize)
    train_limit = 20000
    max_sentence_length = 10
    line_count = 0
    sentence = None
    topCatList = []

    while line_count < len(inputpairs):
        line = inputpairs[line_count]
        line_count += 1
        if line[:5] == "Sent:":
            isQ = False
            donetest=False
            sentence = line[6:].strip().rstrip()
            if sentence.count(" ") > max_sentence_length:
                print("rejecting: example too long ", line)
                sentence = None
                continue
            topCatList = []

        if sentence and line[:4] =="Sem:":
            semstring = line[5:].strip().rstrip()
            try:
                sem, _ = exp.makeExpWithArgs(semstring, {})
            except (AttributeError, IndexError):
                print("LF could not be parsed\nSent : " + sentence)
                print("Sem: " + semstring + "\n\n")
                continue
            if len(sem.allExtractableSubExps()) > 9 and truncate_complex_exps:
                r = None
                sentence = None
                continue

            try:
                isQ, sc = get_top_cat(sem)
            except IndexError:
                # print "couldn't determine syntactic category ", sem_line
                #print_sent_info(sentence, train_out, sentence_count, lexicon, topCatList)
                words = None
                sentence = None
                sem = None
                sc = None
                continue

            words = sentence.split()
            if not isQ and words[-1] in ["?", "."]:
                words = words[:-1]
            if len(words) == 0:
                words = None
                sentence = None
                sem = None
                sc = None
                continue
            # this function is broken atm
            #test_during_training(test_out, sem, words, sem_store, RuleSet, lexicon, sentence_count)

            topCat = cat.cat(sc, sem)
            topCatList.append(topCat)

        if sentence and line[:11] == "example_end":
            print("Sent : " + sentence, file=train_out)
            print("update weight = ", lexicon.get_learning_rate(sentence_count), file=train_out)
            print(sentence_count, file=train_out)
            for topCat in topCatList:
                print("Cat : " + topCat.toString(), file=train_out)
                print("Cat : " + topCat.toString(), file=train_out)

            catStore = {}
            if len(words) > 8 or (skip_q and "?" in sentence):
                sentence = []
                sem = None
                continue

            try:
                chart = build_chart(topCatList, words, RuleSet, lexicon, catStore, sem_store, is_one_word)
            except (AttributeError, IndexError):
                print("Sent : " + sentence)
                continue
            i_o_oneChart(chart, sem_store, lexicon, RuleSet, True, 0.0, sentence_count)
            sentence_count += 1

            if is_devel or (sentence_count == train_limit):
                break
    return lexicon, RuleSet, sem_store, chart

def get_top_cat(sem):
    if sem.checkIfWh():
        isQ = False
        sc = synCat.swh
    elif sem.isQ():
        isQ = True
        sc = synCat.q
    else:
        isQ = False
        sc = synCat.allSynCats(sem.type())[0]
    return isQ, sc

def print_sent_info(sentence, train_out, sentence_count, lexicon, topCatList):
    if topCatList:
        print("sentence is ", sentence)
        print('\ngot training pair')
        print("Sent : " + sentence)
    with open(output_fpath, "w") as f:
        f.write("Sent : " + sentence)
        f.write("update weight = ", lexicon.get_learning_rate(sentence_count))
        f.write(sentence_count)
        if topCatList:
            for topCat in topCatList:
                print("Cat : " + topCat.toString())
                f.write("Cat : " + topCat.toString())
        else:
            print("couldn't determine syntactic category")
            f.write("couldn't determine syntactic category")

def test_during_training(output_fpath, sem, words, sem_store, RuleSet, lexicon, sentence_count):
    (retsem, top_parse, topcat) = parse(words, sem_store, RuleSet, lexicon, sentence_count)
    with open(output_fpath,'w') as f:
        if retsem and sem and retsem.equals(sem):
            f.write("CORRECT\n" + retsem.toString(True) + "\n" + topcat.toString())
        elif not retsem:
            f.write("NO PARSE")
        else:
            f.write("WRONG")
            f.write(retsem.toString(True) + "\n" + topcat.toString())
            f.write(sem.toString(True))
            f.write('top parse:')
            f.write(top_parse)
            f.write("\n")
            if sem and retsem.equalsPlaceholder(sem):
                f.write(f"CORRECTPlaceholder\n{retsem.toString(True)}\n{topcat.toString()}")

def pickle_lexicon(max_lex_dump, sentence_count, min_lex_dump, dump_out, sem_store, lexicon, RuleSet):
    if max_lex_dump >= sentence_count >= min_lex_dump:
       # sentence_count <= max_lex_dump and \
       # sentence_count % dump_interval == 0:
        with open(dump_out + '_' + str(sentence_count), 'wb') as f_lexicon:
            to_pickle_obj = (lexicon, sentence_count, sem_store, RuleSet)
            pickle.dump(to_pickle_obj, f_lexicon, pickle.HIGHEST_PROTOCOL)

def print_top_parse(chart, RuleSet, output_fpath):
    topparses = []
    for entry in chart[len(chart)]:
        top = chart[len(chart)][entry]
        topparses.append((top.inside_score, top))
    top_parse = sample(sorted(topparses)[-1][1], chart, RuleSet)

    with open(output_fpath, "w") as f:
        f.write('top parse:')
        print(top_parse,file=f)
        print(top.inside_score,file=f)

def print_cat_probs(cats_to_check, lexicon, sem_store, RuleSet):
    print("outputting cat probs")
    # this samples the probabilities of each of the syn cat for a given type
    for c in cats_to_check:
        posType = c[0]
        lfType = c[2]
        arity = c[3]
        # these go in cats
        cats = c[4]
        outputFile = c[1]
        outputCatProbs(posType, lfType, arity, cats, lexicon, sem_store, RuleSet, outputFile)

def generate_sentences(sentstogen, lexicon, RuleSet, catStore, sem_store, is_one_word, genoutfile, sentence_count):
    sentnum = 1
    for (gensent, gensemstr) in sentstogen:
        gensem = exp.makeExpWithArgs(gensemstr, {})[0]
        if gensem.checkIfWh():
            sc = synCat.swh
        elif gensem.isQ():
            sc = synCat.q
        else:
            sc = synCat.allSynCats(gensem.type())[0]
        genCat = cat.cat(sc, gensem)
        print("gonna generate sentence ", gensent)
        generateSent(lexicon, RuleSet, genCat, catStore, sem_store, is_one_word, gensent, genoutfile,
                     sentence_count, sentnum)
        sentnum += 1

def test(test_in, test_out, errors_out, sem_store, RuleSet, Current_Lex, sentence_count):
    Current_Lex.refresh_all_params(sentence_count)
    retsem = None
    for line in test_in:
        if line[:5] == "Sent:":
            sentence = line[6:].split()
        if line[:4] == "Sem:":
            try:
                sem = exp.makeExpWithArgs(line[5:].strip().rstrip(), {})[0]
            except IndexError:
                print(sentence, file=errors_out)
                print(line, file=errors_out)
                continue
                #sem = exp.makeExpWithArgs(line[5:].strip().rstrip(), {})[0]
            if not sem.isQ() and sentence[-1] in [".", "?"]:
                sentence = sentence[:-1]
            if len(sentence) == 0:
                sem = None
                sentence = None
                continue
            print(sentence, file=test_out)
            retsem = None
            top_parse = None
            try:
                (retsem, top_parse, topcat) = parse(sentence, sem_store, RuleSet, Current_Lex, sentence_count, test_out)
            except (AttributeError, IndexError):
                pass
            if retsem and sem and retsem.equals(sem):
                print("CORRECT\n" + retsem.toString(True) + "\n" + topcat.toString(), file=test_out)

            elif not retsem:
                print("NO PARSE", file=test_out)
                continue
            else:
                print("WRONG", file=test_out)
                print(retsem.toString(True) + "\n" + topcat.toString(), file=test_out)
                print(sem.toString(True), file=test_out)
                if sem and retsem.equalsPlaceholder(sem):
                    print("CORRECTPlaceholder\n" + retsem.toString(True) + "\n" + topcat.toString(), file=test_out)

            print('top parse:', file=test_out)
            print(top_parse, file=test_out)
            print("\n", file=test_out)
