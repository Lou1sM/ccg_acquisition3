import re

all_consts = set()
for i in range(1,19):
    f = "final_outputs_new_distractors/train_train_parses_reps1__1W_"+str(i)
    p = re.compile("[\,\.\(\)][^\,\.\(\)\|]+\|[^\(,\.]+\(")
    prev_line = None
    for line in open(f,"r"):
        line = line.strip()
        if line.startswith("Cat :"):
            allConsts = [x[1:-1] for x in p.findall(line)]
            for w in allConsts:
                print(prev_line+' '+w)
                if w not in all_consts:
                    all_consts.add(w)
        prev_line = line



