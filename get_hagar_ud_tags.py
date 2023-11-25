#from dtw import dtw
from copy import copy
from config import pos_marking_dict
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

print(len(conll), len(ida_lf))

ida_sents = [x.split('\n')[0][6:].strip() for x in ida_lf]
ida_sents_ = [x.rstrip(' .?!') for x in ida_sents]
ida_lfs = [x.split('\n')[1][5:].strip() for x in ida_lf]
conll_sents = [' '.join(parse_conll(c)[1]) for c in conll] # take lemmas bc they're what's in sent
out_lfs = []
out_sents = []
all_missings = set()

ques = []
cms = []
excluded_for_underscore = 0
excluded_for_bab = 0
chil_counts = {}
all_chilts = {}
for sent,ilf in zip(ida_sents, ida_lfs):
    orig_ilf = ilf
    orig_sent = copy(sent)
    corr_conll = conll[conll_sents.index(sent)]
    assert len(sent.split())== len(corr_conll.split('\n'))
    word_forms, lemmas, chiltags = parse_conll(corr_conll)
    #chiltags = [x.split('\t')[4] for x in corr_conll.split('\n')]
    assert ' '.join(lemmas) == sent
    #chil_dict = dict(zip(sent.split(),chiltags))
    lemma_chil_dict = dict(zip(lemmas,chiltags))
    #lemma_chil_dict = {k.lower():v for k,v in zip(lemmas,chiltags)}
    if any(c not in list(pos_marking_dict.keys())+['.','cm','!','?','pro:person'] for c in chiltags):
        ques += [k for k,v in lemma_chil_dict.items() if v=='que']
        continue
    if 'hagār' in ilf or 'hagāri' in ilf:
        continue
    word_chil_dict = dict(zip(word_forms,chiltags))
    lemma_word_dict = dict(zip(lemmas,word_forms))
    word_lemma_dict = dict(zip(word_forms,lemmas))
    lemma_chil_dict = {k:'pro:per' if v=='pro:person' else v for k,v in lemma_chil_dict.items()}
    word_chil_dict = {k:'pro:per' if v=='pro:person' else v for k,v in word_chil_dict.items()}
    #chil_dict.update(lemma_chil_dict)
    ilf_ =  re.sub(lrs[1:],'',ilf)
    appearing_lf_atoms = set(re.sub(r'(\$\d{1,2}|\(|\))', ' ', ilf_).split())
    missings = [x for x in appearing_lf_atoms if x not in word_lemma_dict and x not in ['and','BARE','not','Q','_']]
    ilf_parts = [x for x in re.split(r'[,.()]',ilf) if x not in ['you','', 'and', 'not', 'BARE', 'Q',''] and '$' not in x]
    if any('$' not in p and p not in ['you','', 'and', 'not', 'BARE', 'Q'] + list(word_chil_dict.keys()) for p in ilf_parts):
        print([p for p in ilf_parts if '$' not in p and p not in ['you','', 'and', 'not', 'BARE', 'Q'] + list(word_chil_dict.keys())])
        continue
    for k,v in lemma_word_dict.items():
        sent = sent.replace(k,v)
    replace_coords = [(orig_ilf.index(x),len(x)) for x in ilf_parts]
    #for ilfp in ilf_parts:
    assert ( ilf.replace('BARE','').replace('Q','').islower())
    for wf in sorted(word_forms, key=lambda x:len(x),reverse=True):
        if wf not in [',', '.']:
            lemma = word_lemma_dict[wf]
            chil = word_chil_dict[wf]
            ilf = ilf.replace(wf.lower(), f'{chil}|{lemma}'.upper())
    ilf = ilf.lower().replace('bare','BARE')
    ilf = re.sub(r'\bq\b','Q',ilf)
    #for loc, size in replace_coords:
    #    head, to_replace, tail = orig_ilf[:loc], orig_ilf[loc:loc+size], orig_ilf[loc+size:]
    #    chilt = word_chil_dict[to_replace]
    #    lemma = word_lemma_dict[to_replace]
    #    ilf = head + f'{chilt}|{lemma}' + tail
    sent = sent.replace(', ','')
    out_sents.append(sent)
    for v in set(lemma_chil_dict.values()):
        chil_counts[v] = chil_counts.get(v,0) + 1
    for k,v in set(lemma_chil_dict.items()):
        all_chilts[v] = all_chilts.get(v,[]) + [k]
    if '_' in lemma_chil_dict.values():
        excluded_for_underscore+=1
        continue
    assert 'bab' not in lemma_chil_dict.values()
    if 'ʔōpro:dem' in ilf:
        breakpoint()
    for m in missings:
        all_missings.add(m)
    assert all('lambda' not in x for x in missings)
    assert all(x not in sent.split() for x in missings)
    print(f'{orig_ilf} --> {ilf}')
    if 'cm' in sent.split():
        cms.append(corr_conll)
    out_lfs.append(ilf)

print(chil_counts)
print(set(ques))
all_chilts = {k:list(set(v)) for k,v in all_chilts.items()}
with open('all_hagar_chilts.json','w') as f:
    json.dump(all_chilts, f)
with open('cms.txt','w') as f:
    print(cms,file=f)
with open('ques.txt','w') as f:
    print(list(set(ques)),file=f)
with open('data/hagar_comma_format.txt','w') as f:
    for sent,olf in zip(out_sents, out_lfs):
        f.write(f'Sent: {sent}\nSem: {olf}\nexample_end\n\n')

all_missings = sorted(all_missings)
breakpoint()
with open('hagar_lf_atoms_not_in_sents.txt','w') as f:
    for m in all_missings:
        f.write(m+'\n')

