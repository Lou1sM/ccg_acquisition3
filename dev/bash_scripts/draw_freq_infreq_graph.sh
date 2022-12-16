set dir = $1

foreach reps (1 3 5 7 )

set files = `ls ${dir}/*reps${reps}*.verb_repo -v | tr '\n' ':'`

python get_verb_stats.py ${files} x___freq_infreq_reps${reps}.png > log_fr_infr_reps${reps}

end
