import json
import numpy as np
import argparse

ARGS = argparse.ArgumentParser()
ARGS.add_argument("--num_dpoints", type=int, default=1000)
ARGS.add_argument("--spaces", action="store_true")
ARGS.add_argument("--use_nouns", action="store_true")
ARGS.add_argument("--with_questions", action="store_true")
ARGS = ARGS.parse_args()

with open('data/preprocessed_geoqueries.json') as f:
    d = json.load(f)

nps = d['np_list']

transitives = ['like', 'buy', 'has', 'address', 'border', 'hate', 'want', 'throw', 'traverse', 'devour', 'kick', 'love', 'enrich','deracinate']

nouns = ['state', 'capital', 'place', 'lake', 'mountain', 'city', 'river','dog','cat','table','person','apple']
intransitives = ['run', 'walk', 'jump', 'talk', 'sing', 'play', 'read','write','arrive','exist','remain','leave','dance','confabulate','prevaricate','dead_lift']

nums = ['one','two','three','four','five','six','seven','eight','nine','ten']

def coin_flip(p=0.5):
    return np.random.rand()<p

def NP_with_determiner():
    det = np.random.choice(['the','a'])
    noun = np.random.choice(nouns)
    lf = f'({det} (lambda $0.{noun} $0))'
    return f'{det} {noun}', lf

def name_NP():
    subj = np.random.choice(nps)
    return subj, subj

def make_sentence(is_q):
    subj_words, subj_lf = NP_with_determiner() if coin_flip() else name_NP()
    verb = np.random.choice(list(transitives) + intransitives)
    verb_words = verb if is_q else verb+'s'
    if verb in intransitives:
        VP = verb_words
        declarative_lf = f'{verb} {subj_lf}'
    else:
        obj_words, obj_lf = NP_with_determiner() if coin_flip() else name_NP()
        VP = f'{verb_words} {obj_words}'
        declarative_lf = f'{verb} {subj_lf} {obj_lf}'
    words = f'does {subj_words} {VP}' if is_q else f'{subj_words} {VP}'
    lf = f'Q({declarative_lf})' if is_q else declarative_lf
    return {'words':words.replace('_',' ').split(), 'lf':lf}

def make_relative():
    base_sent, base_lf = make_sentence().values()
    subj = np.random.choice(nps)
    verb = np.random.choice(transitives)
    words = f'{base_sent} which {subj} {intransitives[verb]}'
    lf = f'AND ({base_lf}) ({subj} {verb} {base_lf.split()[-1]})'
    return {'words':words.replace('_',' ').split(), 'lf':lf}

#dpoints = [make_simple_sentence(np.random.rand() > 0.5 and ARGS.use_nouns) for _ in range(ARGS.num_dpoints)]
dpoints = [make_sentence(coin_flip(0.5 if ARGS.with_questions else 0)) for _ in range(ARGS.num_dpoints)]
processed_dset = {'np_list':nps, 'nouns':nouns, 'intransitive_verbs':list(intransitives),
                    'transitive_verbs': list(transitives), 'data':dpoints}
breakpoint()
print(processed_dset)
fpath = f'data/determiners_questions{ARGS.num_dpoints}.json' if ARGS.with_questions else f'data/determiners{ARGS.num_dpoints}.json'
with open(fpath,'w') as f:
    json.dump(processed_dset,f)
