import json
import numpy as np
import argparse

ARGS = argparse.ArgumentParser()
ARGS.add_argument("--num_dpoints", type=int, default=1000)
ARGS.add_argument("--spaces", action="store_true")
ARGS.add_argument("--use_nouns", action="store_true")
ARGS = ARGS.parse_args()

with open('data/preprocessed_geoqueries.json') as f:
    d = json.load(f)

nps = d['np_list']

transitives = {'longer': ('is_longer_than', False), 'lower': ('is_lower_than', False), 'low_point': ('is_a_low_point_of', False), 'area': ('has_area', True), 'capital_of': ('is_the_captial_of', False), 'next_to': ('is_next_to', False), 'len': ('has_length', True), 'population': ('has_population', True), 'size': ('has_size', True), 'traverse': ('traverses', False), 'higher': ('is_higher_than', False), 'high_point': ('is_a_high_point_of', False), 'density': ('has_density', True), 'elevation': ('has_elevation', True), 'loc': ('is_in', False)}

#intransitives = {'state': 'is_a_state', 'capital': 'is_a_capital', 'place': 'is_a_place', 'lake': 'is_a_lake', 'mountain': 'is_a_mountain', 'city': 'is_a_city', 'river': 'is_a_river'}
nouns = ['state', 'capital', 'place', 'lake', 'mountain', 'city', 'river']
intransitives = ['runs', 'walks', 'jumps', 'talks', 'sings', 'plays', 'reads','writes']

if ARGS.spaces:
    transitives = {k:(v.replace('_',' '),b) for k,(v,b) in transitives.items()}
    intransitives = {k:v.replace('_',' ') for k,v in intransitives.items()}

nums = ['one','two','three','four','five','six','seven','eight','nine','ten']

def NP_with_determiner():
    det = np.random.choice(['the','a'])
    noun = np.random.choice(nouns)
    lf = f'({det} (lambda $0.{noun} $0))'
    return f'{det} {noun}', lf

def name_NP():
    subj = np.random.choice(nps)
    return subj, subj

def make_sentence():
    subj_words, subj_lf = NP_with_determiner() if np.random.rand()>0.5 else name_NP()
    verb = np.random.choice(list(transitives) + intransitives)
    if verb in intransitives:
        words = f'{subj_words} {verb}'
        lf = f'{verb} {subj_lf}'
    else:
        verb_words,requires_num = transitives[verb]
        if requires_num:
            obj_words = np.random.choice(nums)
            obj_lf = str(nums.index(obj_words)+1) # because of zero-indexing
        else:
            obj_words, obj_lf = NP_with_determiner() if np.random.rand()>0.5 else name_NP()
        words = f'{subj_words} {verb_words} {obj_words}'
        lf = f'{verb} {subj_lf} {obj_lf}'
    return {'words':words.replace('_',' ').split(), 'lf':lf}

def make_relative():
    base_sent, base_lf = make_simple_sentence()
    subj = np.random.choice(nps)
    verb = np.random.choice(transitives)
    words = f'{base} which {subj} {intransitives[verb]}'
    lf = f'AND ({base_lf}) ({subj} {verb} {base_lf.split()[-1]})'

#dpoints = [make_simple_sentence(np.random.rand() > 0.5 and ARGS.use_nouns) for _ in range(ARGS.num_dpoints)]
dpoints = [make_sentence() for _ in range(ARGS.num_dpoints)]
processed_dset = {'np_list':nps, 'nouns':nouns, 'intransitive_verbs':list(intransitives),
                    'transitive_verbs': list(transitives), 'data':dpoints}
fpath = f'data/determiners_spaces{ARGS.num_dpoints}.json'# if ARGS.spaces else f'data/{base}{ARGS.num_dpoints}.json'
with open(fpath,'w') as f:
    json.dump(processed_dset,f)
