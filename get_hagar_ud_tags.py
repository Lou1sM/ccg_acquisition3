#from dtw import dtw
import numpy as np
import json
import re
from utils import LAMBDA_RE_STR as lrs


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
conll_sents = [' '.join([x.split()[1] for x in c.split('\n')]) for c in conll]
out_lfs = []
all_missings = set()

for sent,ilf,mlf in zip(ida_sents, ida_lfs, my_lfs):
    corr_conll = conll[conll_sents.index(sent)]
    assert len(sent.split())== len(corr_conll.split('\n'))
    ud_dict = dict(zip(sent.split(),[x.split('\t')[4] for x in corr_conll.split('\n')]))
    for k,v in ud_dict.items():
        if k!='.':
            ilf = ilf.replace(k,f'{k}|{v}')
    mlf_ =  re.sub(lrs[1:],'',mlf)
    appearing_lf_atoms = set(re.sub(r'(\$\d{1,2}|\(|\))', ' ', mlf_).split())
    missings = [x for x in appearing_lf_atoms if x not in ud_dict and x not in ['and','BARE','not','Q','_']]
    for m in missings:
        all_missings.add(m)
    assert all('lambda' not in x for x in missings)
    assert all(x not in sent.split() for x in missings)
    out_lfs.append(ilf)

with open('data/hagar_comma_format.txt','w') as f:
    for sent,olf in zip(ida_sents, out_lfs):
        f.write(f'Sent: {sent}\nSem: {olf}\nexample_end\n\n')

all_missings = sorted(all_missings)
with open('hagar_lf_atoms_not_in_sents.txt','w') as f:
    for m in all_missings:
        f.write(m+'\n')

