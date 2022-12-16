
set dir = $1
set output_dir = $2
set high_res_dir = $3

# draw accuracy graphs
#set acc_dist1 = `ls ${dir}/train_test_parses_reps1__1W_* -v | tr '\n' ':'`
#set acc_dist3 = `ls ${dir}/train_test_parses_reps3__1W_* -v | tr '\n' ':'`
#set acc_dist5 = `ls ${dir}/train_test_parses_reps5__1W_* -v | tr '\n' ':'`
#set acc_dist7 = `ls ${dir}/train_test_parses_reps7__1W_* -v | tr '\n' ':'`

#python eval_test_file.py ${output_dir}/acc_graphs $acc_dist1 $acc_dist3 $acc_dist5 $acc_dist7

# draw transitive verbs

#foreach i (1 3 5 7)
foreach i (1)
   set trans_cats = `ls ${dir}/*reps${i}*.trans_cats  -v | tr '\n' ':'`
   python draw_syn_cats_graph.py ${trans_cats} ${output_dir}/trans_reps${i}_syn_6way.png #"Learned Prior for Transitive Verbs"
end


# draw naturalistic

#foreach reps ( 1 3 5 7 )

#foreach phenom ( trans prep det nouns )

#python draw_syn_w_given_lf_graph.py -d ${dir}/ -p ${phenom} -c reps${reps} -s out -o ${output_dir}/${phenom}_reps${reps}

#end

#end


# draw dax

#foreach reps (1 3 5 7 )

#set phenom = trans_dax

#python draw_syn_w_given_lf_graph.py -d ${dir}/ -p ${phenom} -c reps${reps} -s out1 -o ${output_dir}/${phenom}_reps${reps}_1_ 

#python draw_syn_w_given_lf_graph.py -d ${dir}/ -p ${phenom} -c reps${reps} -s out2 -o ${output_dir}/${phenom}_reps${reps}_2_ 

#set phenom = corp_prep

#python draw_syn_w_given_lf_graph.py -d ${dir}/ -p ${phenom} -c reps${reps} -s out3 -o ${output_dir}/${phenom}_reps${reps}_ 

#set phenom = corp_noun

#python draw_syn_w_given_lf_graph.py -d ${dir}/ -p ${phenom} -c reps${reps} -s out4 -o ${output_dir}/${phenom}_reps${reps}_

# frequent vs. infrequent

#set files = `ls ${dir}/*reps${reps}*.verb_repo -v | tr '\n' ':'`

#python get_verb_stats.py ${files} ${output_dir}/freq_infreq_reps${reps}.png

#end



# draw lexical explosion

#set files = `ls ${dir}/train_test_parses_reps7_dump__*.out -v | tr '\n' ':'`

#python eval_lexical_explosion.py $files 0.8 ${output_dir}/lex_explosion.png



# draw nouns vs. verbs

#foreach reps ( 1 3 5 7 )

#set files = `ls ${high_res_dir}/train_test_parses_reps${reps}_dump__*.out -v | tr '\n' ':'`

#python eval_verb_vs_noun.py $files 0.8 ${output_dir}/verbs_vs_nouns_abs${reps}.png F

#end


