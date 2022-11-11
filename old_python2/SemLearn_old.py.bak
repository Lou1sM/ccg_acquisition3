#############################################
## This is getting stripped down so that I ##
## can see what the hell is going on.      ##
#############################################
## this needs to be sorted out so that I can do things with it
## want high level options for hyperparameters
## want high level option for one vs many words
## want ONE function to check parses (deal with search problem)
## MAKE INTO SEPARATE FILES

import cPickle
import sys
import verb_repo
from optparse import OptionParser
from generator import generateSent
from grammar_classes import *
from lexicon_classes import *
from parser import *
from sample_most_probable_parse import *
from makeGraphs import *
from cat import synCat
import cat
import extract_from_lexicon3
import exp
import expFunctions


noQ = False


def train_rules(sem_store, RuleSet, lexicon, oneWord, inputpairs,
                cats_to_check, output, test_out=None, dotest=False, sentence_count=0,
                min_lex_dump=0, max_lex_dump=1000000, dump_lexicons=False,
                dump_interval=100, dump_out='lexicon_dump', f_out_additional=None, truncate_complex_exps=True,
                verb_repository=None, dump_verb_repo=False, analyze_lexicons=False, genoutfile=None):
    print "testout = ", test_out
    print "put in sent coutn = ", sentence_count
    datasize = 10000
    lexicon.set_learning_rates(datasize)
    # wordstocheck = []
    # wordstocheck.extend(['the','a','some','another','any'])
    # wordstocheck.extend(['the','a','some','another','an','any','all','both'])
    # wordstocheck.extend(['spoon','cup','pencil','name','coffee','lady','drink','duck'])
    # wordstocheck.extend(['give','bring','find','write','tell'])
    # wordstocheck.extend(['dancing','sit','went','blow'])
    # wordstocheck.extend(['doing','put','have','want','like','say','take','dropped','open','peel','pull','fall'])
    # wordstocheck.extend(['now','better','just','later','again','already','upstairs','first','almost'])
    # sentstogen = [("I ate a cookie","lambda $0_{ev}.PAST(v|eat(pro|I,det|a($1,n|cookie($1)),$0))")]
    #                  ],\
    #                      ("I took it","lambda $0_{ev}.v|take&PAST(pro|I,pro|it,$0)"),\
    #                      ("more juice ?","lambda $0_{ev}.Q(qn|more($1,n|juice($1)),$0)"),\
    #                      ("drink the water","lambda $0_{ev}.v|drink(pro|you,det|the($1,n|water($1)),$0)"),
    #                  ("he doesn 't have a hat",
    # "lambda $0_{ev}.not(aux|do&3S(v|have(pro|he,det|a($1,n|hat($1)),$0),$0),$0)")]
    # lexicon.set_words_to_check(wordstocheck)

    sentstogen = []
    train_limit = 10000
    max_sentence_length = 10
    line_count = 0
    sentence = None
    topCatList = []

    with open("./trainFiles/Adam_troublesome_lf.txt", "a") as failed_out:
        while line_count < len(inputpairs):
            # line = inputpairs[line_count]
            # line_count += 1
            # if line[:5] == "Sent:":
            #     isQ = False
            #     donetest=False
            #     sentence = line[6:].strip().rstrip()
            #     if sentence.count(" ") > max_sentence_length:
            #         print "rejecting: example too long ", line
            #         sentence = None
            #         continue
            #     topCatList = []
            #
            # if sentence and line[:4] =="Sem:":
            #     semstring = line[5:].strip().rstrip()
            #     try:
            #         sem, _ = expFunctions.makeExpWithArgs(semstring, {})
            #     except (AttributeError, IndexError):
            #         print >> output, "LF could not be parsed\nSent : " + sentence
            #         print >> output, "Sem: " + semstring + "\n\n"
            #         continue
            #     if len(sem.allExtractableSubExps()) > 9 and truncate_complex_exps:
            #         print "rejecting ", sem.toString(True)
            #         r = None
            #         sentence = None
            #         continue
            #
            #     try:
            #         isQ, sc = get_top_cat(sem)
            #     except IndexError:
            #         # print "couldn't determine syntactic category ", sem_line
            #         print_sent_info(sentence, output, sentence_count, lexicon, topCatList)
            #         continue
            #
            #     words = sentence.split()
            #     if not isQ and words[-1] in ["?", "."]:
            #         words = words[:-1]
            #     else:
            #         print "Is Q"
            #     if len(words) == 0:
            #         words = None
            #         sentence = None
            #         sem = None
            #         sc = None
            #         continue
            #     if dotest and not donetest:
            #         test_during_training(test_out, sem, words, sem_store, RuleSet, lexicon, sentence_count)
            #
            #     print "sentence is ", sentence
            #     topCat = cat.cat(sc, sem)
            #     topCatList.append(topCat)
            #
            #
            # if sentence and line[:11] == "example_end":
            #     print '\ngot training pair'
            #     print "Sent : " + sentence
            #     if sentence == "where 's the real Ursula ?":
            #         print("booo")
            #         pass
            #     print >> output, "Sent : " + sentence
            #     print >> output, "update weight = ", lexicon.get_learning_rate(sentence_count)
            #     print >> output, sentence_count
            #     for topCat in topCatList:
            #         print "Cat : " + topCat.toString()
            #         print >> output, "Cat : " + topCat.toString()
            #
            #     catStore = {}
            #     if len(words) > 8 or (noQ and "?" in sentence):
            #         sentence = []
            #         sem = None
            #         continue
            #
            #     try:
            #         chart = build_chart(topCatList, words, RuleSet, lexicon, catStore, sem_store, oneWord)
            #     except (AttributeError, IndexError):
            #         print >> failed_out, "Sent : " + sentence
            #         continue
            #     print "got chart"
            #     i_o_oneChart(chart, sem_store, lexicon, RuleSet, True, 0.0, sentence_count)
            #     print "done io"
            #     sentence_count += 1

            sent_line, sem_line, end_line = inputpairs[line_count:line_count+3]
            line_count += 4
            if sent_line[:5] != "Sent:" and sem_line[:4] != "Sem:" and end_line[:11] != "example_end":
                raise AssertionError("wrong triple of lines")
            else:
                sentence = sent_line[6:].strip().rstrip()
                semstring = sem_line[5:].strip().rstrip()
                #######################
                # Read in the example
                #######################
                try:
                    sem, _ = expFunctions.makeExpWithArgs(semstring, {})
                except (AttributeError, IndexError):
                    print >> output, "LF could not be parsed\nSent : " + sentence
                    print >> output, "Sem: " + semstring + "\n\n"
                    continue
                if not sem:
                    raise (StandardError("could not make exp"))
                reject = (sentence.count(" ") > max_sentence_length) or \
                         (len(sem.allExtractableSubExps()) > 9 and truncate_complex_exps)
                if reject:
                    print "rejecting: example too long ", sent_line + sem_line
                    continue

                ########################
                # Get syntactic category
                ########################
                topCatList = []
                try:
                    isQ, sc = get_top_cat(sem)
                except IndexError:
                    # print "couldn't determine syntactic category ", sem_line
                    print_sent_info(sentence, output, sentence_count, lexicon, topCatList)
                    continue
                topCat = cat.cat(sc, sem)
                topCatList.append(topCat)
                print_sent_info(sentence, output, sentence_count, lexicon, topCatList)

                words = sentence.split()
                if not isQ and words[-1] in ["?", "."]:
                    words = words[:-1]
                # if len(words) == 0:
                #     print "rejecting: example too short ", sent_line + sem_line
                #     continue

                if dotest:
                    test_during_training(test_out, sem, words, sem_store, RuleSet, lexicon, sentence_count)

                # print_sent_info(sentence, output, sentence_count, lexicon, topCatList)

                ####################################
                # Create parse forest
                ####################################
                catStore = {}
                try:
                    chart = build_chart(topCatList, words, RuleSet, lexicon, catStore, sem_store, oneWord)
                except (AttributeError, IndexError):
                    print >> failed_out, "Sent : " + sentence
                    continue
                print "got chart"

                ####################################
                # Run inside-outside calculations
                ####################################
                # if chart is not None:
                i_o_oneChart(chart, sem_store, lexicon, RuleSet, True, 0.0, sentence_count)
                print "done io"
                sentence_count += 1

                ####################################
                # Monitor rules, update verb repo,
                # save the model
                ####################################
                watch_selected_rules(lexicon, sem_store, RuleSet, sentence_count, sentence)
                if verb_repository:
                    for cur_cat in set([c.semString() for c in lexicon.cur_cats]):
                        try:
                            verb_repository.add_verb(cur_cat, lexicon, sem_store, \
                                                 RuleSet, sentence_count)
                        except TypeError:
                            #this is not great, I'm putting it in because I don't quite get why the above fails
                            pass
                lexicon.cur_cats = []
                dump_model(dump_lexicons, analyze_lexicons, dump_verb_repo, sentence_count,
                           dump_interval, max_lex_dump, min_lex_dump, dump_out, sem_store,
                           lexicon, RuleSet, verb_repository)
                print_top_parse(chart, RuleSet, output, f_out_additional)
                print_cat_probs(cats_to_check, lexicon, sem_store, RuleSet)

                ####################################
                # Generate sentence from LF
                ####################################
                doingGenerate = False
                if doingGenerate:
                    generate_sentences(sentstogen, lexicon, RuleSet, catStore, sem_store,
                                       oneWord, genoutfile, sentence_count)

                print "done with sent\n\n"
                if sentence_count == train_limit:
                    return sentence_count

        print "returning sentence count ", sentence_count
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

def print_sent_info(sentence, output, sentence_count, lexicon, topCatList):
    if topCatList:
        print "sentence is ", sentence
        print '\ngot training pair'
        print "Sent : " + sentence
    print >> output, "Sent : " + sentence
    print >> output, "update weight = ", lexicon.get_learning_rate(sentence_count)
    print >> output, sentence_count
    if topCatList:
        for topCat in topCatList:
            print "Cat : " + topCat.toString()
            print >> output, "Cat : " + topCat.toString()
    else:
        print "couldn't determine syntactic category"
        print >> output, "couldn't determine syntactic category"

def test_during_training(test_out, sem, words, sem_store, RuleSet, lexicon, sentence_count):
    print >> test_out, "\n****************\n", words
    (retsem, top_parse, topcat) = parse(words, sem_store, RuleSet, lexicon, sentence_count, test_out)
    print >> test_out, "\n", words
    if retsem and sem and retsem.equals(sem):
        print >> test_out, "CORRECT\n" + retsem.toString(True) + "\n" + topcat.toString()
    elif not retsem:
        print >> test_out, "NO PARSE"
    else:
        print >> test_out, "WRONG"
        print >> test_out, retsem.toString(True) + "\n" + topcat.toString()
        print >> test_out, sem.toString(True)

        print >> test_out, 'top parse:'
        print >> test_out, top_parse
        print >> test_out, "\n"
        if sem and retsem.equalsPlaceholder(sem):
            print >> test_out, "CORRECTPlaceholder\n" + retsem.toString(True) + "\n" + topcat.toString()

def pickle_lexicon(max_lex_dump, sentence_count, min_lex_dump, dump_out, sem_store, lexicon, RuleSet):
    if max_lex_dump >= sentence_count >= min_lex_dump:
       # sentence_count <= max_lex_dump and \
       # sentence_count % dump_interval == 0:
        f_lexicon = open(dump_out + '_' + str(sentence_count), 'wb')
        to_pickle_obj = (lexicon, sentence_count, sem_store, RuleSet)
        cPickle.dump(to_pickle_obj, f_lexicon, cPickle.HIGHEST_PROTOCOL)
        f_lexicon.close()

def analyze_lexicon(dump_out, sentence_count, lexicon, sem_store, RuleSet):
    extract_from_lexicon3.main(dump_out + '_' + str(sentence_count) + '.out',
                               lexicon=lexicon, sentence_count=sentence_count,
                               sem_store=sem_store, RuleSet=RuleSet)

def save_verb_repo(dump_out, sentence_count, verb_repository):
    f_repo = open(dump_out + '_' + str(sentence_count) + '.verb_repo', 'wb')
    cPickle.dump(verb_repository, f_repo, cPickle.HIGHEST_PROTOCOL)
    f_repo.close()

def watch_selected_rules(lexicon, sem_store, RuleSet, sentence_count, sentence):
    # added 14/8/2014 for debugging purposes
    target_syn_keys = ["((S\NP)/NP)", "((S/NP)/NP)", "((S\NP)\NP)", "((S/NP)\NP)"]
    syn_distribution = extract_from_lexicon3.get_synt_distribution(target_syn_keys, \
                                                                   lexicon, sem_store, RuleSet,
                                                                   sentence_count)
    print('WATCH' + '\t' + sentence)
    for k, v in syn_distribution.items():
        print('WATCH' + '\t' + str(sentence_count) + '\t' + str(k) + '\t' + str(v))

def print_top_parse(chart, RuleSet, output, f_out_additional):
    print "getting topparses"
    topparses = []
    for entry in chart[len(chart)]:
        top = chart[len(chart)][entry]
        topparses.append((top.inside_score, top))

    top_parse = sample(sorted(topparses)[-1][1], chart, RuleSet)
    print >> output, 'top parse:'
    print >> output, top_parse
    print >> output, top.inside_score
    print >> output, "\n"

    if f_out_additional:
        print >> f_out_additional, '\ntop parse:'
        print >> f_out_additional, top_parse
        print >> f_out_additional, top.inside_score
        print >> f_out_additional, "\n"

def print_cat_probs(cats_to_check, lexicon, sem_store, RuleSet):
    print "outputting cat probs"
    # this samples the probabilities of each of the syn cat for a given type
    for c in cats_to_check:
        posType = c[0]
        lfType = c[2]
        arity = c[3]
        # these go in cats
        cats = c[4]
        outputFile = c[1]
        outputCatProbs(posType, lfType, arity, cats, lexicon, sem_store, RuleSet, outputFile)

def generate_sentences(sentstogen, lexicon, RuleSet, catStore, sem_store, oneWord, genoutfile, sentence_count):
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
        print "gonna generate sentence ", gensent
        generateSent(lexicon, RuleSet, genCat, catStore, sem_store, oneWord, gensent, genoutfile,
                     sentence_count, sentnum)
        sentnum += 1

def dump_model(dump_lexicons, analyze_lexicons, dump_verb_repo, sentence_count, dump_interval,
               max_lex_dump, min_lex_dump, dump_out, sem_store, lexicon, RuleSet, verb_repository):
    # pickling lexicon (added by Omri)
    if dump_lexicons and sentence_count % dump_interval == 0:
        pickle_lexicon(max_lex_dump, sentence_count, min_lex_dump, dump_out, sem_store, lexicon, RuleSet)

    if analyze_lexicons and sentence_count % dump_interval == 0:
        analyze_lexicon(dump_out, sentence_count, lexicon, sem_store, RuleSet)

    if dump_verb_repo and sentence_count % dump_interval == 0:
        save_verb_repo(dump_out, sentence_count, verb_repository)


##########################################################


def test(test_file, sem_store, RuleSet, Current_Lex, test_out, sentence_count):
    errors_out = open("my_test/sem_errors.txt", "a")
    Current_Lex.refresh_all_params(sentence_count)
    retsem = None
    for line in test_file:
        if line[:5] == "Sent:":
            sentence = line[6:].split()
        if line[:4] == "Sem:":
            try:
                sem = expFunctions.makeExpWithArgs(line[5:].strip().rstrip(), {})[0]
            except IndexError:
                print >> errors_out, sentence
                print >> errors_out, line
                continue
                #sem = expFunctions.makeExpWithArgs(line[5:].strip().rstrip(), {})[0]
            if not sem.isQ() and sentence[-1] in [".", "?"]:
                sentence = sentence[:-1]
            if len(sentence) == 0:
                sem = None
                sentence = None
                continue
            print >> test_out, sentence
            retsem = None
            top_parse = None
            try:
                (retsem, top_parse, topcat) = parse(sentence, sem_store, RuleSet, Current_Lex, sentence_count, test_out)
            except (AttributeError, IndexError):
                pass
            if retsem and sem and retsem.equals(sem):
                print >> test_out, "CORRECT\n" + retsem.toString(True) + "\n" + topcat.toString()

            elif not retsem:
                print >> test_out, "NO PARSE"
                continue
            else:
                print >> test_out, "WRONG"
                print >> test_out, retsem.toString(True) + "\n" + topcat.toString()
                print >> test_out, sem.toString(True)
                if sem and retsem.equalsPlaceholder(sem):
                    print >> test_out, "CORRECTPlaceholder\n" + retsem.toString(True) + "\n" + topcat.toString()

            print >> test_out, 'top parse:'
            print >> test_out, top_parse
            print >> test_out, "\n"


###########################################
# Main.                                   #
# Try to keep to just build or check      #
###########################################

def main(argv, options):
    print argv
    build_or_check = argv[1]
    if len(argv) > 2:
        To = argv[2]
    else:
        To = None
    if len(argv) > 3:
        k = argv[3]
    else:
        k = None
    print "build or check is ", build_or_check

    if build_or_check == "i_n":  # train on months 1..i, test on the n-th
        exp.exp.allowTypeRaise = False

        # initialization info #
        oneWord = True
        if len(argv) > 2 and argv[2] in ["mwe", "MWE"]:
            oneWord = False
        numreps = 1
        if len(argv) > 3:
            numreps = int(argv[3])
            print('Number of possible LFs in training:' + str(numreps))
        if len(argv) > 4:
            extra = argv[4]
        else:
            extra = ""

        reverse = False  # True
        if reverse: extra = extra + "reversed"

        Lexicon.set_one_word(oneWord)

        rule_alpha_top = 1.0
        beta_tot = 1.0
        beta_lex = 0.005

        verb_repository = verb_repo.VerbRepository()
        RuleSet = Rules(rule_alpha_top, beta_tot, beta_lex)

        type_to_shell_alpha_o = 1000.0
        shell_to_sem_alpha_o = 500.0
        word_alpha_o = 1.0

        Current_Lex = Lexicon(type_to_shell_alpha_o, shell_to_sem_alpha_o, word_alpha_o)

        RuleSet.usegamma = False
        Current_Lex.usegamma = False

        sentence_count = 0

        cats_to_check = []

        sem_store = SemStore()
        test_file_index = options.test_session
        # if options.development_mode:
        #     test_file_index = 19
        # else:
        #     test_file_index = 2

        # if options.pickle_model:
        #     pickle_file = "_".join(options.test_parses.split("_")[-3:-1])+".pkl"

        if options.continued:
            dump_file = open(options.continued, "rb")
            model_dict = cPickle.load(dump_file)
            sem_store = model_dict["sem_store"]
            RuleSet = model_dict["RuleSet"]
            Current_Lex = model_dict["Current_Lex"]
            sentence_count = model_dict["sentence_count"]
            cats_to_check = model_dict["cats_to_check"]
            # Lexicon = model_dict["Lexicon"]
            # rule_alpha_top = model_dict["rule_alpha_top"]
            # beta_tot = model_dict["beta_tot"]
            # beta_lex = model_dict["beta_lex"]
            # type_to_shell_alpha_o = model_dict["type_to_shell_alpha_o"]
            # shell_to_sem_alpha_o = model_dict["shell_to_sem_alpha_o"]
            # word_alpha_o = model_dict["word_alpha_o"]
            start_file = model_dict["last_session_no"] + 1
        else:
            start_file = 1

        for i in range(start_file, test_file_index):
            # input_file = options.inp_file.format(str(i))
            # test_file = options.inp_file.format(test_file_index)
            input_file = options.inp_file
            test_file = options.inp_file+"_"+str(test_file_index)

            if numreps > 1:
                input_file = input_file + str(numreps) + "reps"
            input_file = input_file + "_" + str(i)

            inputpairs = open(input_file).readlines()

            #outfile = options.train_parses.format(str(i)) + '_'
            #testoutfile = options.test_parses.format(str(i)) + '_'
            outfile = options.train_parses + '_'
            testoutfile = options.test_parses + '_'

            if oneWord:
                outfile = outfile + "1W"
                testoutfile = testoutfile + "1W"
            else:
                outfile = outfile + "MWE"
                testoutfile = testoutfile + "MWE"

            if numreps > 1:
                outfile = outfile + "_" + str(numreps) + "reps"
                testoutfile = testoutfile + "_" + str(numreps) + "reps"

            outfile = outfile + "_" + str(i)
            testoutfile = testoutfile + "_" + str(i)
            output = open(outfile, "w")

            sentence_count = train_rules(sem_store, RuleSet, Current_Lex, oneWord, inputpairs,
                                         cats_to_check, output, None, False , sentence_count,
                                         min_lex_dump=options.min_lex_dump,
                                         max_lex_dump=options.max_lex_dump,
                                         dump_lexicons=options.dump_lexicons,
                                         dump_interval=options.dump_interval,
                                         dump_out=options.dump_out,
                                         verb_repository=verb_repository,
                                         dump_verb_repo=options.dump_verb_repo,
                                         analyze_lexicons=options.analyze_lexicons)

            print "returned sentence count = ", sentence_count

            dotest = options.dotest
            if dotest:
                test_out = open(testoutfile, "w")
                print >> test_out, "trained on up to ", input_file, " testing on ", test_file
                test_file = open(test_file, "r")
                test(test_file, sem_store, RuleSet, Current_Lex, test_out, sentence_count)
                test_out.close()
            if options.pickle_model:
                pickle_prefix = "_".join(options.test_parses.split("_")[-3:-1]) + "_"+str(i)

                f = open(pickle_prefix+"_sem_store", "wb")
                cPickle.dump(sem_store, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()
                print("\n\n\n**************************\nPICKLED sem_store\n**************************\n")
                f2 = open(pickle_prefix + "_RuleSet", "wb")
                cPickle.dump(RuleSet, f2, protocol=cPickle.HIGHEST_PROTOCOL)
                f2.close()
                print("\n\n\n**************************\nPICKLED RuleSet\n**************************\n")
                f3 = open(pickle_prefix + "_Current_Lex", "wb")
                cPickle.dump(Current_Lex, f3, protocol=cPickle.HIGHEST_PROTOCOL)
                f3.close()
                print("\n\n\n**************************\nPICKLED Current_Lex\n**************************\n")
                f4 = open(pickle_prefix + "_cats_to_check", "wb")
                cPickle.dump(cats_to_check, f4, protocol=cPickle.HIGHEST_PROTOCOL)
                f4.close()
                print("\n\n\n**************************\nPICKLED cats_to_check\n**************************\n")
                f5 = open(pickle_prefix + "_sentence_count", "wb")
                cPickle.dump(sentence_count, f5, protocol=cPickle.HIGHEST_PROTOCOL)
                f5.close()
                print("\n\n\n**************************\nPICKLED sentence count\n**************************\n")
                f6 = open(pickle_prefix + "_session_no", "wb")
                cPickle.dump(i, f6, protocol=cPickle.HIGHEST_PROTOCOL)
                f6.close()
                print("\n\n\n**************************\nPICKLED session no\n**************************\n")

                # dict_to_pickle = {"sem_store": sem_store,
                #                   "RuleSet": RuleSet,
                #                   "Current_Lex": Current_Lex,
                #                   "sentence_count": sentence_count,
                #                   "cats_to_check": cats_to_check,
                #                   # "Lexicon": Lexicon,
                #                   # "rule_alpha_top": rule_alpha_top,
                #                   # "beta_tot": beta_tot,
                #                   # "beta_lex": beta_lex,
                #                   # "type_to_shell_alpha_o": type_to_shell_alpha_o,
                #                   # "shell_to_sem_alpha_o": shell_to_sem_alpha_o,
                #                   # "word_alpha_o": word_alpha_o,
                #                   "last_session_no": i}
                # f = open(pickle_file, "wb")
                # cPickle.dump(dict_to_pickle, f)
                # f.close()
        print "at end, lexicon size is ", len(Current_Lex.lex)


def cmd_line_parser():
    """
    Returns the command line parser.
    """
    usage = "usage: %prog [options]\n"
    opt_parser = OptionParser(usage=usage)
    opt_parser.add_option("--min_lex_dump", action="store", type="int", dest="min_lex_dump", default=0,
                          help="the number of iterations before we start dumping lexicons")
    opt_parser.add_option("--max_lex_dump", action="store", type="int", dest="max_lex_dump", default=1000000,
                          help="the number of iterations at which we stop dumping lexicons")
    opt_parser.add_option("-d", action="store_true", dest="dump_lexicons", default=False,
                          help="whether to dump the lexicons or not")
    opt_parser.add_option("--ai", action="store", dest="alternative_input",
                          help="an alternative input file (works only with load_from_pickle)")
    opt_parser.add_option("--dl", action="store", dest="dumped_lexicon",
                          help="a dumped lexicon file (works only with load_from_pickle")
    opt_parser.add_option("--dotest", action="store_true", dest="dotest", default=False,
                          help="use this flag if you want to apply testing")
    opt_parser.add_option("--dinter", action="store", dest="dump_interval", type="int", default=100,
                          help="a dumped lexicon file (works only with load_from_pickle")
    opt_parser.add_option("--dump_out", action="store", dest="dump_out", default='lexicon_dump',
                          help="the prefix for the lexicon dumps")
    opt_parser.add_option("-t", action="store", dest="test_parses",
                          help="the output file for the test parses")
    opt_parser.add_option("-n", action="store", dest="train_parses",
                          help="the output file for the train parses")
    opt_parser.add_option("--dump_vr", action="store_true", dest="dump_verb_repo", default=False,
                          help="whether to dump the verb repository")
    opt_parser.add_option("-i", dest="inp_file", default="trainFiles/trainPairs",
                          help="the input file names (with the annotated corpus)")
    opt_parser.add_option("--analyze", dest="analyze_lexicons", default=False, action="store_true",
                          help="output the results for the experiments")
    # opt_parser.add_option("-k", dest="learning_rate_power",default=None,type="float",
    #                      help="power of the learning rate")
    opt_parser.add_option("--devel", dest="development_mode", default=False, action="store_true",
                          help="development mode")
    opt_parser.add_option("-s", dest="test_session", default=26, action="store", type="int",
                          help="number of session on which to test; training up to the previous session")
    opt_parser.add_option("--p", dest="pickle_model", default=False, action="store_true",
                          help="whether to save the model after every training session")
    opt_parser.add_option("-c", dest="continued", default=None,
                          help="pickled learner files to use as as the initial state")

    return opt_parser


if __name__ == '__main__':
    parser = cmd_line_parser()
    options, args = parser.parse_args(sys.argv)
    if len(args) == 1 or args[1] != 'i_n':
        print('Illegal option for now.')
        sys.exit(-1)
    if not options.train_parses or not options.test_parses:
        print('Train and test parses have to be defined')
        sys.exit(-1)

    main(args, options)
