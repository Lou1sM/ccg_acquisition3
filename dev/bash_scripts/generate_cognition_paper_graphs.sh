csh draw_all.sh final_outputs_new_distractors final_graphs final_outputs_new_distractors_high_res

set i = 3
set dir = final_outputs_new_distractors_k-0.6.old
set trans_cats = `ls ${dir}/*reps${i}*.trans_cats  -v | tr '\n' ':'`

python draw_syn_cats_graph.py ${trans_cats} final_graphs/trans_reps${i}_syn_6way_gradLR.png

csh draw_all_1000utts.sh final_outputs_new_distractors_high_res final_graphs

