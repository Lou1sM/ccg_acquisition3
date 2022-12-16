set dir = $1

foreach reps ( 1 3 5 7 )

foreach phenom ( trans prep det nouns )

python draw_syn_w_given_lf_graph.py -d ${dir}/ -p ${phenom} -c reps${reps} -s out -o ${dir}/${phenom}_reps${reps}

end

end






