foreach i (1 3 5 7)
   set trans_cats = `ls final_outputs_new_distractors/*reps${i}*.trans_cats  -v | tr '\n' ':'`
   python draw_syn_cats_graph.py ${trans_cats} ~/trans_reps${i}_syn.6way.png
end

