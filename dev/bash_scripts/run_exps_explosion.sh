set dir = final_outputs_new_distractors

set distractors = 7

python -u SemLearn.py i_n - ${distractors} -i trainFiles/trainPairs.filtered.ready -t ${dir}/xx -n ${dir}/x --dump_vr --dinter 20 --analyze --dump_out ${dir}/train_test_parses_reps${distractors}_dump_ --dotest >& log_lex_explosion & 


