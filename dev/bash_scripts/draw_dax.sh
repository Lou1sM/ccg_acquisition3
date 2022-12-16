foreach reps (1 3 7 )

set phenom = trans_dax

python draw_syn_w_given_lf_graph.py -d newest_outputs/ -p ${phenom} -c reps${reps} -s out1 -o final_outputs_graphs/${phenom}_reps${reps}_1_ &

python draw_syn_w_given_lf_graph.py -d newest_outputs/ -p ${phenom} -c reps${reps} -s out2 -o final_outputs_graphs/${phenom}_reps${reps}_2_ &

set phenom = corp_prep

python draw_syn_w_given_lf_graph.py -d newest_outputs/ -p ${phenom} -c reps${reps} -s out3 -o final_outputs_graphs/${phenom}_reps${reps}_ &

set phenom = corp_noun

python draw_syn_w_given_lf_graph.py -d newest_outputs/ -p ${phenom} -c reps${reps} -s out4 -o final_outputs_graphs/${phenom}_reps${reps}_ &

end


