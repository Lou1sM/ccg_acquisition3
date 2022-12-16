set dir  = $1

foreach i (1 3 5 7)
   set trans_cats = `ls ${dir}/*reps${i}*.trans_cats  -v | tr '\n' ':'`
   python draw_syn_cats_graph.py ${trans_cats} ${dir}/trans_reps${i}_syn.6way.png
end
