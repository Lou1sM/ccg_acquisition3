import json
import os
import numpy as np
from numpy.random import choice
import argparse


def coin_flip(p=0.5):
    return np.random.rand()<p

def NP_with_determiner(is_plural:bool):
    det = choice(['the','a','my','your'])
    noun = choice(nouns)
    #lf = f'({det} (lambda $0.{noun} $0))'
    if is_plural:
        noun += '-s'
        if det == 'a':
            det = 'the'
    lf = f'({det} {noun})' # these brackets are the only diff between _words and _lf, but easier to reason about
    return f'{det} {noun}', lf

def name_NP():
    subj = choice(nps)
    return subj, subj

def make_NP():
    if ARGS.determiners and coin_flip():
        is_plural = coin_flip(0)
        subj_words, subj_lf = NP_with_determiner(is_plural)
    else:
        subj_words, subj_lf = name_NP()
        is_plural = False
    return subj_words, subj_lf, is_plural

def get_VP():
    verb = choice(list(transitive_verbs) + intransitive_verbs)
    #verb_words = verb if is_q or subj_words.endswith('-s') else verb+'s'
    if verb in intransitive_verbs:
        obj_words = obj_lf = None
    else:
        obj_words, obj_lf, _ = make_NP()
    #if verb in intransitive_verbs:
        #VP = verb_words
        #declarative_lf = f'{verb} {subj_lf}'
    #else:
        #obj_words, obj_lf = NP_with_determiner(False) if coin_flip() else name_NP()
        #VP = f'{verb_words} {obj_words}'
        #declarative_lf = f'{verb} {obj_lf} {subj_lf}'
    #return VP, declarative_lf
    return verb, obj_words, obj_lf

def make_S():
    is_q = ARGS.questions and coin_flip()
    subj_words, subj_lf, is_plural = make_NP()
    verb, obj_words, obj_lf = get_VP()
    verb_words = verb if is_q or subj_words.endswith('-s') else verb+'s'
    S_lf = ' '.join(filter(None,[verb,subj_lf,obj_lf]))
    S_words = ' '.join(filter(None,[subj_words,verb_words,obj_words]))
    if obj_lf == subj_lf:
        obj_lf = obj_lf+'_x'
    if is_q:
        do_word = 'do' if is_plural else 'does'
        words = f'{do_word} {S_words}'
        lf = f'Q ({S_lf})'
    else:
        words = S_words
        lf = S_lf
    return {'words':words.replace('_',' ').split(), 'lf':lf}

def make_control_sentence():
    subj_words, subj_lf, is_plural = make_NP()
    verb, obj_words, obj_lf = get_VP()
    cv = choice(control_verbs)

    VP_lf = ' '.join(filter(None,[verb,subj_lf,obj_lf]))
    lf = f'{cv} {subj_lf} ({VP_lf})'

    VP_words = ' '.join(filter(None,[verb,obj_words]))
    cv_words = cv if is_plural else cv+'-s'
    words = f'{subj_words} {cv_words} to {VP_words}'

    print({'words':words.replace('_',' ').split(), 'lf':lf})
    return {'words':words.replace('_',' ').split(), 'lf':lf}

def make_relative_sentence():
    base_sent, base_lf = make_S().values()
    subj = choice(nps)
    verb = choice(transitive_verbs)
    words = f'{base_sent} which {subj} {intransitive_verbs[verb]}'
    lf = f'AND ({base_lf}) ({subj} {verb} {base_lf.split()[-1]})'
    return {'words':words.replace('_',' ').split(), 'lf':lf}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_dpoints", type=int, default=1000)
    data_properties_group = parser.add_argument_group('data properties')
    data_properties_group.add_argument("--control", action="store_true")
    data_properties_group.add_argument("--determiners", action="store_true")
    data_properties_group.add_argument("--overwrite", action="store_true")
    data_properties_group.add_argument("--print_dset", action="store_true")
    data_properties_group.add_argument("--questions", action="store_true")
    data_properties_group.add_argument("--relatives", action="store_true")

    ARGS = parser.parse_args()

    potential_properties = ['control','determiners','questions','relatives']
    data_properties = [x for x in potential_properties if getattr(ARGS,x)]
    sentence_type_funcs = [make_S]
    if ARGS.control:
        sentence_type_funcs.append(make_control_sentence)
    if ARGS.relatives:
        sentence_type_funcs.append(make_relative_sentence)

    with open('data/preprocessed_geoqueries.json') as f:
        d = json.load(f)

    nps = d['np_list'] + ['you']

    transitive_verbs = ['like', 'buy', 'has', 'address', 'border', 'hate', 'want', 'throw', 'traverse', 'devour', 'kick', 'love', 'enrich','deracinate']

    nouns = ['state', 'capital', 'place', 'lake', 'mountain', 'city', 'river','dog','cat','table','person','apple']
    intransitive_verbs = ['run', 'walk', 'jump', 'talk', 'sing', 'play', 'read','write','arrive','exist','remain','leave','dance','confabulate','prevaricate','dead_lift']
    control_verbs = ['try','hope','plan','fail']

    nums = ['one','two','three','four','five','six','seven','eight','nine','ten']

    dpoints = [choice(sentence_type_funcs)() for _ in range(ARGS.n_dpoints)]
    processed_dset = {'np_list':nps, 'nouns':nouns, 'intransitive_verbs':list(intransitive_verbs),
                        'transitive_verbs': list(transitive_verbs), 'data':dpoints}
    if ARGS.print_dset:
        print(processed_dset)
    fpath = f'data/{"_".join(data_properties)}{ARGS.n_dpoints}.json'
    if os.path.exists(fpath) and not ARGS.overwrite:
        fpath += '.1'
        print(f'path already exists, renaming to {fpath}')
    with open(fpath,'w') as f:
        json.dump(processed_dset,f)
    print('saved to',fpath)
