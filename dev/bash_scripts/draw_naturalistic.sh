foreach reps ( 1 3 7 )

foreach phenom ( trans prep det nouns )

python draw_syn_w_given_lf_graph.py -d final_outputs/ -p ${phenom} -c reps${reps} -s out -o final_outputs_graphs/${phenom}_reps${reps}

end

end


