import os
import verb_repo
import argparse
from lexicon_classes import Lexicon
from grammar_classes import Rules
from build_inside_outside_chart import SemStore
import pickle
from sem_learn import train_rules, test
#from grammar_classes import *
#from parser import *
#from sample_most_probable_parse import *
#from makeGraphs import *


def main(args):
    rule_alpha_top = 1.0
    beta_tot = 1.0
    beta_lex = 0.005

    verb_repository = verb_repo.VerbRepository()
    RuleSet = Rules(rule_alpha_top, beta_tot, beta_lex)

    type_to_shell_alpha_o = 1000.0
    shell_to_sem_alpha_o = 500.0
    word_alpha_o = 1.0

    lexicon = Lexicon(type_to_shell_alpha_o, shell_to_sem_alpha_o, word_alpha_o, args.include_mwe)

    RuleSet.usegamma = False
    lexicon.usegamma = False

    sentence_count = 0

    cats_to_check = []

    sem_store = SemStore()
    test_file_index = args.test_session
    if args.pickle_model:
        pickle_file = "_".join(args.test_parses.split("_")[-4:-1])+".pkl"

    if args.continued:
        dump_file = open(args.continued, "rb")
        model_dict = pickle.load(dump_file)
        sem_store = model_dict["sem_store"]
        RuleSet = model_dict["RuleSet"]
        lexicon = model_dict["Current_Lex"]
        sentence_count = model_dict["sentence_count"]
        cats_to_check = model_dict["cats_to_check"]
        start_file = model_dict["last_session_no"] + 1
    else:
        start_file = 1

    for i in range(start_file, test_file_index):
        input_file = args.inp_file
        test_file = args.inp_file+"_"+str(test_file_index)

        if args.numreps > 1:
            input_file = input_file + str(args.numreps) + "reps"
        input_file = input_file + "_" + str(i)

        inputpairs = open(input_file).readlines()

        train_parses_fpath = os.path.join(args.outdir, 'train_parses' + str(i) + '.txt')
        test_parses_fpath = os.path.join(args.outdir, 'test_parses.' + str(i) + 'txt')

        sentence_count = train_rules(sem_store, RuleSet, lexicon, args.include_mwe, inputpairs,args.skip_q,
                                     cats_to_check, train_parses_fpath, None, False, sentence_count,
                                     min_lex_dump=args.min_lex_dump,
                                     max_lex_dump=args.max_lex_dump,
                                     dump_lexicons=args.dump_lexicons,
                                     dump_interval=args.dump_interval,
                                     dump_out=args.dump_out,
                                     verb_repository=verb_repository,
                                     is_dump_verb_repo=args.is_dump_verb_repo,
                                     analyze_lexicons=args.is_analyze_lexicons)


        print("returned sentence count = ", sentence_count)

        if args.dotest:
            test_file = open(test_file, "r")
            test(test_parses_fpath, sem_store, RuleSet, lexicon, sentence_count)
        if args.pickle_model:
            dict_to_pickle = {"sem_store": sem_store,
                              "RuleSet": RuleSet,
                              "Current_Lex": lexicon,
                              "sentence_count": sentence_count,
                              "cats_to_check": cats_to_check,
                              "last_session_no": i}
            f = open(pickle_file, "wb")
            pickle.dump(dict_to_pickle, f)
            f.close()
        print("at end, lexicon size is ", len(lexicon.lex))


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--min_lex_dump", type=int, default=0,
                          help="the number of iterations before we start dumping lexicons")
    args.add_argument("--numreps", type=int, default=1)
    args.add_argument("--outdir", type=str, default="tmp")
    args.add_argument("--dump_out", type=str, default="lexicon_dump")
    args.add_argument("--max_lex_dump", type=int, default=1000000,
                          help="the number of iterations at which we stop dumping lexicons")
    args.add_argument("-d", "--dump_lexicons", action="store_true",
                          help="whether to dump the lexicons or not")
    args.add_argument("--dotest", action="store_true",
                          help="use this flag if you want to apply testing")
    args.add_argument("--dump_interval", type=int, default=100,
                          help="a dumped lexicon file (works only with load_from_pickle")
    args.add_argument("--expname", default='tmp',
                          help="the directory to write output files")
    args.add_argument("--is_dump_verb_repo", action="store_true",
                          help="whether to dump the verb repository")
    args.add_argument("-i", "--inp_file", default="trainFiles/trainPairs",
                          help="the input file names (with the annotated corpus)")
    args.add_argument("--is_analyze_lexicons", action="store_true",
                          help="output the results for the experiments")
    args.add_argument("--devel", "--development_mode", action="store_true",
                          help="development mode")
    args.add_argument("-s", "--test_session", default=26, type=int,
                          help="number of session on which to test; training up to the previous session")
    args.add_argument("-p", "--pickle_model", action="store_true",
                          help="whether to save the model after every training session")
    args.add_argument("--skip_q", action="store_true")
    args.add_argument("-c", "--continued", default=None,
                          help="pickled learner files to use as as the initial state")
    args.add_argument("--include_mwe", action="store_true")
    args = args.parse_args()

    main(args)
