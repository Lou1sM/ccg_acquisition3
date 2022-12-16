
set distractors = 7

python SemLearn.py i_n - ${distractors} --dump_vr --dump_out verb_repo/reps${distractors}_ -t xx${distractors} -n xx${distractors} >& log_verb_repo${distractors} &



