import extract_from_lexicon, sys

MAX = {1: 5000 , 3: 4100, 7: 2800}
num_reps = sys.argv[1]
start_index = int(sys.argv[2])

LEXICON_FILES = \
    ['newest_outputs/lex_dump_reps'+num_reps+'__'+str(j) for j in range(start_index,MAX[int(num_reps)]+100,100)]
for lexicon_fn in LEXICON_FILES:
    print(lexicon_fn)
    extract_from_lexicon.main(lexicon_fn, lexicon_fn+'.output', 'F')


