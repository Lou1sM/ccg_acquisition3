
set dir = $1

foreach reps ( 1 3 5 7 )

set files = `ls ${dir}/train_test_parses_reps${reps}_dump__*.out -v | tr '\n' ':'`

python eval_verb_vs_noun.py $files 0.8 $dir/verbs_vs_nouns_abs${reps}.png F

end




