
set dir = $1
set output_dir = $2

# draw naturalistic

foreach reps ( 1 3 5 7 )

set phenom = trans

python draw_syn_w_given_lf_graph.py -d ${dir}/ -p ${phenom} -c reps${reps} -s out -o ${output_dir}/${phenom}_reps${reps}_occ --xlim 2000 --occ

foreach phenom ( prep det nouns )

python draw_syn_w_given_lf_graph.py -d ${dir}/ -p ${phenom} -c reps${reps} -s out -o ${output_dir}/${phenom}_reps${reps}_occ --xlim 1000 --occ

end

end



