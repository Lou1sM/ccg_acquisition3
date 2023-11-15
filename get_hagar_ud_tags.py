#from dtw import dtw
import numpy as np
import json
import re
from utils import LAMBDA_RE_STR as lrs


def parse_conll(conll):
    split_lines = [x.split('\t') for x in conll.split('\n')]
    word_forms = [x[1] for x in split_lines]
    lemmas = [x[2] for x in split_lines]
    chiltags = [x[4] for x in split_lines]
    return word_forms, lemmas, chiltags

with open('../CHILDES_UD2LF/conll/full_hagar/hagar.all.udv1.conllu.current') as f:
    conll = f.read().strip().split('\n\n')

with open('../CHILDES_UD2LF/LF_files/full_hagar/hagar.all_lf.txt') as f:
    ida_lf = f.read().strip().split('\n\n')

with open('data/simplified_hagar.json') as f:
    d=json.load(f)['data']

my_sents = [' '.join(x['words']) for x in d]
my_lfs = [x['lf'] for x in d]
print(len(conll), len(ida_lf))

ida_sents = [x.split('\n')[0][6:].strip() for x in ida_lf]
ida_sents_ = [x.rstrip(' .?!') for x in ida_sents]
assert all(x in ida_sents_ for x in my_sents)
ida_lfs = [x.split('\n')[1][5:].strip() for x in ida_lf]
conll_sents = [' '.join(parse_conll(c)[1]) for c in conll] # take lemmas bc they're what's in sent
out_lfs = []
out_sents = []
all_missings = set()

for sent,ilf,mlf in zip(ida_sents, ida_lfs, my_lfs):
    corr_conll = conll[conll_sents.index(sent)]
    assert len(sent.split())== len(corr_conll.split('\n'))
    word_forms, lemmas, chiltags = parse_conll(corr_conll)
    #chiltags = [x.split('\t')[4] for x in corr_conll.split('\n')]
    assert ' '.join(lemmas) == sent
    #chil_dict = dict(zip(sent.split(),chiltags))
    lemma_chil_dict = dict(zip(lemmas,chiltags))
    word_chil_dict = dict(zip(word_forms,chiltags))
    lemma_word_dict = dict(zip(lemmas,word_forms))
    word_lemma_dict = dict(zip(word_forms,lemmas))
    lemma_chil_dict = {k:'pro:per' if v=='pro:person' else v for k,v in lemma_chil_dict.items()}
    word_chil_dict = {k:'pro:per' if v=='pro:person' else v for k,v in word_chil_dict.items()}
    #chil_dict.update(lemma_chil_dict)
    ilf_ =  re.sub(lrs[1:],'',ilf)
    appearing_lf_atoms = set(re.sub(r'(\$\d{1,2}|\(|\))', ' ', ilf_).split())
    missings = [x for x in appearing_lf_atoms if x not in word_lemma_dict and x not in ['and','BARE','not','Q','_']]
    for k,v in word_lemma_dict.items():
        chilt = word_chil_dict[k]
        ilf = ilf.replace(k,f'{chilt}|{v}')
        sent = sent.replace(v,k)
    sent = sent.replace(', ','')
    out_sents.append(sent)
    for m in missings:
        all_missings.add(m)
    if not all('lambda' not in x for x in missings):
        breakpoint()
    if not all(x not in sent.split() for x in missings):
        breakpoint()
    out_lfs.append(ilf)

with open('data/hagar_comma_format.txt','w') as f:
    for sent,olf in zip(out_sents, out_lfs):
        f.write(f'Sent: {sent}\nSem: {olf}\nexample_end\n\n')

all_missings = sorted(all_missings)
with open('hagar_lf_atoms_not_in_sents.txt','w') as f:
    for m in all_missings:
        f.write(m+'\n')

