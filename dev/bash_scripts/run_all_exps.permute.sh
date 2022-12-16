
set dir = final_outputs_new_distractors_no_questions

set distractors = 1

python -u SemLearn.py i_n -i trainFiles/trainPairs.filtered.ready.permute -t ${dir}/train_test_parses_reps${distractors}_ -n ${dir}/train_train_parses_reps${distractors}_ --dump_out ${dir}/train_test_parses_reps${distractors}_dump_ --dump_vr --dinter 100 --analyze --dotest >& log_${distractors}rep_dump_permute_${dir} & 


foreach distractors ( 3 5 7 )

python -u SemLearn.py i_n - ${distractors} -i trainFiles/trainPairs.filtered.ready.permute -t ${dir}/train_test_parses_reps${distractors}_ -n ${dir}/train_train_parses_reps${distractors}_ --dump_vr --dinter 100 --analyze  --dump_out ${dir}/train_test_parses_reps${distractors}_dump_ --dotest >& log_${distractors}rep_dump_permute_${dir} & 

end




