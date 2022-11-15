import pickle
import verb_repo
from generator import generateSent
from grammar_classes import *
from inside_outside_calc import i_o_oneChart
from parser import *
from sample_most_probable_parse import *
from makeGraphs import *
from cat import synCat
import cat
import extract_from_lexicon3
import exp
import expFunctions


def train_rules(sem_store, RuleSet, lexicon, is_one_word, inputpairs, skip_q,
                cats_to_check, output_fpath, test_out=None, dotest=False, sentence_count=0,
                min_lex_dump=0, max_lex_dump=1000000, dump_lexicons=False,
                dump_interval=100, dump_out='lexicon_dump', f_out_additional=None, truncate_complex_exps=True,
                verb_repository=None, is_dump_verb_repo=False, analyze_lexicons=False, genoutfile=None):
    datasize = 20000
    lexicon.set_learning_rates(datasize)

    sentstogen = []
    train_limit = 20000
    max_sentence_length = 10
    line_count = 0
    sentence = None
    topCatList = []

    #with open("./trainFiles/Adam_troublesome_lf.txt", "a") as failed_out:
    with open(output_fpath, "w") as f:
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
                    sem, _ = expFunctions.makeExpWithArgs(semstring, {})
                except (AttributeError, IndexError):
                    f.write("LF could not be parsed\nSent : " + sentence)
                    f.write("Sem: " + semstring + "\n\n")
                    continue
                if len(sem.allExtractableSubExps()) > 9 and truncate_complex_exps:
                    print("rejecting ", sem.toString(True))
                    r = None
                    sentence = None
                    continue

                try:
                    isQ, sc = get_top_cat(sem)
                except IndexError:
                    # print "couldn't determine syntactic category ", sem_line
                    #print_sent_info(sentence, train_parses_fpath, sentence_count, lexicon, topCatList)
                    words = None
                    sentence = None
                    sem = None
                    sc = None
                    continue

                words = sentence.split()
                if not isQ and words[-1] in ["?", "."]:
                    words = words[:-1]
                else:
                    print("Is Q")
                if len(words) == 0:
                    words = None
                    sentence = None
                    sem = None
                    sc = None
                    continue
                if dotest and not donetest:
                    test_during_training(test_out, sem, words, sem_store, RuleSet, lexicon, sentence_count)

                topCat = cat.cat(sc, sem)
                topCatList.append(topCat)

            if sentence and line[:11] == "example_end":
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
                print("got chart")
                i_o_oneChart(chart, sem_store, lexicon, RuleSet, True, 0.0, sentence_count)
                print("done io")
                sentence_count += 1

                if verb_repository:
                    for cur_cat in set([c.semString() for c in lexicon.cur_cats]):
                        try:
                            verb_repository.add_verb(cur_cat, lexicon, sem_store, \
                                                 RuleSet, sentence_count)
                        except TypeError:
                            breakpoint()
                            #this is not great, I'm putting it in because I don't quite get why the above fails
                            pass
                lexicon.cur_cats = []
                dump_model(dump_lexicons, analyze_lexicons, is_dump_verb_repo, sentence_count,
                           dump_interval, max_lex_dump, min_lex_dump, dump_out, sem_store,
                           lexicon, RuleSet, verb_repository)
                #anothe hack - not sure why printing a parse would fail if a chart has been created but it does sometimes
                #try:
                    #print_top_parse(chart, RuleSet, train_parses_fpath, f_out_additional)
                #except IndexError:
                    #print("Top parse priting failed")
                try:
                    print_cat_probs(cats_to_check, lexicon, sem_store, RuleSet)
                except Exception:
                    print("Who knows, but something failed")

                ####################################
                # Generate sentence from LF
                ####################################
                doingGenerate = False
                if doingGenerate:
                    generate_sentences(sentstogen, lexicon, RuleSet, catStore, sem_store,
                                       is_one_word, genoutfile, sentence_count)

                if sentence_count == train_limit:
                    return sentence_count
    return sentence_count

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

def print_sent_info(sentence, train_parses_fpath, sentence_count, lexicon, topCatList):
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

def test_during_training(test_out, sem, words, sem_store, RuleSet, lexicon, sentence_count):
    (retsem, top_parse, topcat) = parse(words, sem_store, RuleSet, lexicon, sentence_count, test_out)
    if retsem and sem and retsem.equals(sem):
        print("CORRECT\n" + retsem.toString(True) + "\n" + topcat.toString(), file=test_out)
    elif not retsem:
        print("NO PARSE", file=test_out)
    else:
        print("WRONG", file=test_out)
        print(retsem.toString(True) + "\n" + topcat.toString(), file=test_out)
        print(sem.toString(True), file=test_out)

        print('top parse:', file=test_out)
        print(top_parse, file=test_out)
        print("\n", file=test_out)
        if sem and retsem.equalsPlaceholder(sem):
            print("CORRECTPlaceholder\n" + retsem.toString(True) + "\n" + topcat.toString(), file=test_out)

def pickle_lexicon(max_lex_dump, sentence_count, min_lex_dump, dump_out, sem_store, lexicon, RuleSet):
    if max_lex_dump >= sentence_count >= min_lex_dump:
       # sentence_count <= max_lex_dump and \
       # sentence_count % dump_interval == 0:
        f_lexicon = open(dump_out + '_' + str(sentence_count), 'wb')
        to_pickle_obj = (lexicon, sentence_count, sem_store, RuleSet)
        pickle.dump(to_pickle_obj, f_lexicon, pickle.HIGHEST_PROTOCOL)
        f_lexicon.close()

def analyze_lexicon(dump_out, sentence_count, lexicon, sem_store, RuleSet):
    lex_log = open("lexicon_log.txt", "a")
    lex_log.write("/n".join(list(lexicon.sem_distribution.sem_to_pairs.keys())))
    lex_log.write("#### next session ####")
    extract_from_lexicon3.main(dump_out + '_' + str(sentence_count) + '.out',
                               lexicon=lexicon, sentence_count=sentence_count,
                               sem_store=sem_store, RuleSet=RuleSet)

def save_verb_repo(dump_out, sentence_count, verb_repository):
    f_repo = open(dump_out + '_' + str(sentence_count) + '.verb_repo', 'wb')
    pickle.dump(verb_repository, f_repo, pickle.HIGHEST_PROTOCOL)
    f_repo.close()

def print_top_parse(chart, RuleSet, train_parses_fpath, f_out_additional):
    print("getting topparses")
    topparses = []
    for entry in chart[len(chart)]:
        top = chart[len(chart)][entry]
        topparses.append((top.inside_score, top))

    top_parse = sample(sorted(topparses)[-1][1], chart, RuleSet)
    with open(output_fpath, "w") as f:
        f.write('top parse:')
        f.write(top_parse)
        f.write(top.inside_score)
        f.write("\n")

    if f_out_additional:
        print('\ntop parse:', file=f_out_additional)
        print(top_parse, file=f_out_additional)
        print(top.inside_score, file=f_out_additional)
        print("\n", file=f_out_additional)

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
        gensem = expFunctions.makeExpWithArgs(gensemstr, {})[0]
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

def dump_model(dump_lexicons, analyze_lexicons, is_dump_verb_repo, sentence_count, dump_interval,
               max_lex_dump, min_lex_dump, dump_out, sem_store, lexicon, RuleSet, verb_repository):
    # pickling lexicon (added by Omri)
    if dump_lexicons and sentence_count % dump_interval == 0:
        pickle_lexicon(max_lex_dump, sentence_count, min_lex_dump, dump_out, sem_store, lexicon, RuleSet)

    if analyze_lexicons and sentence_count % dump_interval == 0:
        analyze_lexicon(dump_out, sentence_count, lexicon, sem_store, RuleSet)

    if is_dump_verb_repo and sentence_count % dump_interval == 0:
        save_verb_repo(dump_out, sentence_count, verb_repository)

def test(exp_name, sem_store, RuleSet, Current_Lex, sentence_count):
    errors_out_fpath = os.path.join(exp_name,'sem_errors.txt')
    errors_out = open(errors_out_fpath, "a")
    Current_Lex.refresh_all_params(sentence_count)
    retsem = None
    for line in test_file:
        if line[:5] == "Sent:":
            sentence = line[6:].split()
        if line[:4] == "Sem:":
            try:
                sem = expFunctions.makeExpWithArgs(line[5:].strip().rstrip(), {})[0]
            except IndexError:
                print(sentence, file=errors_out)
                print(line, file=errors_out)
                continue
                #sem = expFunctions.makeExpWithArgs(line[5:].strip().rstrip(), {})[0]
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


