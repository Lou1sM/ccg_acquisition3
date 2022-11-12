#python -u SemLearn.py i_n -i trainFiles/trainPairs.adam.ready -t my_test/train_test_parses_adam_reps1_ -n my_test/train_train_parses_adam_reps1_ --dump_vr --dinter 100 --analyze -s 41 --dump_out my_test/train_test_parses_Adam_reps1_dump_ --dotest
python SemLearn.py -i trainFiles/trainPairs.adam.ready -t --dump_vr --dinter 100 --analyze -s 41 --dump_out --dotest
python SemLearn.py -i trainFiles/trainPairs.adam.ready --is_dump_verb_repo --is_analyze_lexicon -s 41 --dotest
