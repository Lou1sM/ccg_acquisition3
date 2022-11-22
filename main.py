import os
import verb_repo
import argparse
from lexicon_classes import Lexicon
from grammar_classes import Rules
from build_inside_outside_chart import SemStore
import pickle
from train_test import train_rules, test, print_cat_probs, print_top_parse
from extract_from_lexicon import extract_from_lexicon


def main(args):
    rule_alpha_top = 1.0
    beta_tot = 1.0
    beta_lex = 0.005

    rule_set = Rules(rule_alpha_top, beta_tot, beta_lex)

    type_to_shell_alpha_o = 1000.0
    shell_to_sem_alpha_o = 500.0
    word_alpha_o = 1.0

    sentence_count = 0

    lexicon = Lexicon(type_to_shell_alpha_o, shell_to_sem_alpha_o, word_alpha_o, args.include_mwe)
    cats_to_check = []

    sem_store = SemStore()

    if args.pickle_model:
        pickle_file = "_".join(args.test_parses.split("_")[-4:-1])+".pkl"

    if args.continued:
        dump_file = open(args.continued, "rb")
        model_dict = pickle.load(dump_file)
        sem_store = model_dict["sem_store"]
        rule_set = model_dict["rule_set"]
        lexicon = model_dict["Current_Lex"]
        sentence_count = model_dict["sentence_count"]
        cats_to_check = model_dict["cats_to_check"]
        start_file = model_dict["last_session_no"] + 1
    else:
        start_file = 1

    for i in range(start_file, args.test_session):
        with open(f'data/{args.corpus}.{i}') as f:
            inputpairs = f.readlines()
        train_out_fpath = os.path.join(args.outdir, f'train_out{i}.txt')
        test_out_fpath = os.path.join(args.outdir, f'test_out{i}.txt')

        with (  open(train_out_fpath, 'w') as train_out,
                open(test_out_fpath, 'w') as test_out):
            lexicon, rule_set, sem_store, chart = train_rules(lexicon,rule_set,sem_store,
                                    is_one_word=not args.include_mwe, inputpairs=inputpairs,
                                    skip_q=args.skip_q,
                                    cats_to_check=cats_to_check,sentence_count=sentence_count,
                                    train_out=train_out,
                                    test_out=test_out,
                                    is_devel=args.devel)

        lexicon.cur_cats = []
        print_cat_probs(cats_to_check, lexicon, sem_store, rule_set)
        print_top_parse(chart, rule_set, train_out_fpath)

        if args.is_dump_lexicons:
            dump_lexicons_fpath = os.path.join(args.outdir, f'lexicon_dump{i}.txt')
            with open(dump_lexicons_fpath, 'wb') as f_lexicon:
                to_pickle_obj = (lexicon, sentence_count, sem_store, rule_set)
                pickle.dump(to_pickle_obj, f_lexicon, pickle.HIGHEST_PROTOCOL)

        if args.is_analyze_lexicons:
            log_lexicons_fpath = os.path.join(args.outdir, f'lexicon_log{i}.txt')
            with open(log_lexicons_fpath,'a') as f:
                f.write("/n".join(list(lexicon.sem_distribution.sem_to_pairs.keys())))
                f.write("#### next session ####")
            lexicon_analysis_fpath = os.path.join(args.outdir, f'lexicon_analysis{i}.txt')
            extract_from_lexicon(lexicon_analysis_fpath, lexicon,
                            sem_store, rule_set, sentence_count=sentence_count)

        if args.is_dump_verb_repo:
            verb_repository = verb_repo.VerbRepository()
            for cur_cat in set([c.semString() for c in lexicon.cur_cats]):
                verb_repository.add_verb(cur_cat, lexicon, sem_store, rule_set, sentence_count)
            verb_repo_dump_fpath = os.path.join(args.outdir, f'verb_repo_dump{i}.txt')
            with open(verb_repo_dump_fpath,'wb') as f:
                pickle.dump(verb_repository, f, pickle.HIGHEST_PROTOCOL)

        if args.pickle_model:
            dict_to_pickle = {"sem_store": sem_store,
                              "rule_set": rule_set,
                              "Current_Lex": lexicon,
                              "sentence_count": sentence_count,
                              "cats_to_check": cats_to_check,
                              "last_session_no": i}
            with open(pickle_file, "wb") as f:
                pickle.dump(dict_to_pickle, f)

    # end for
    if args.dotest:
        test_in_fpath = f'data/{args.corpus}.{args.test_session}'
        test_out_fpath = os.path.join(args.outdir, f'test_out.txt')
        errors_out_fpath = os.path.join(args.outdir, f'errors_out.txt')
        with (  open(test_in_fpath, 'r') as test_in,
                open(test_out_fpath, 'w') as test_out,
                open(errors_out_fpath, 'w') as errors_out):
            test(test_in, test_out, errors_out, sem_store, rule_set, lexicon, sentence_count)

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
    args.add_argument("--is_dump_lexicons", action="store_true",
                          help="whether to dump the lexicons or not")
    args.add_argument("--dotest", action="store_true",
                          help="use this flag if you want to apply testing")
    args.add_argument("--dump_interval", type=int, default=100,
                          help="a dumped lexicon file (works only with load_from_pickle")
    args.add_argument("--expname", default='tmp',
                          help="the directory to write output files")
    args.add_argument("--is_dump_verb_repo", action="store_true",
                          help="whether to dump the verb repository")
    args.add_argument("--corpus", default="adam",help="can specify reps, e.g adam.5reps")
    args.add_argument("--is_analyze_lexicons", action="store_true")
    args.add_argument("--is_generate_sentences", action="store_true",
                          help="doesn't seem to be used atm")
    args.add_argument("--devel", "--development_mode", action="store_true")
    args.add_argument("-s", "--test_session", default=26, type=int,
                          help="session on which to test; train up to the previous session")
    args.add_argument("-p", "--pickle_model", action="store_true",
                          help="whether to save the model after every training session")
    args.add_argument("--skip_q", action="store_true")
    args.add_argument("-c", "--continued", default=None,
                          help="pickled learner files to use as as the initial state")
    args.add_argument("--include_mwe", action="store_true")
    args = args.parse_args()

    if args.devel:
        args.test_session = 2
    main(args)
