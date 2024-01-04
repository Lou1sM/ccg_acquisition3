import pandas as pd
import re
import numpy as np


with open('data/adam_comma_format.txt') as f:
    alflines = f.readlines()
with open('data/hagar_comma_format.txt') as f:
    hlflines = f.readlines()

with open('../CHILDES_UD2LF/conll/full_adam/adam.all.udv1.conllu.final') as f:
    aconlines = f.readlines()
with open('../CHILDES_UD2LF/conll/full_hagar/hagar.all.udv1.conllu.current') as f:
    hconlines = f.readlines()

def lf_terms(lf):
    return lf.replace('.',',').replace(')',',').replace('(',',').split(',')

def get_novs(dlines):
    novs_list = []
    for dl in dlines:
        if dl.startswith('Sem:'):
            terms = lf_terms(dl)
            if not any(t.startswith('v|') for t in terms):
                novs_list.append(dl)
    return novs_list

anovs = get_novs(alflines)
hnovs = get_novs(hlflines)
breakpoint()

def counts_from_lf_lines(dlines,dset_name):
    counts = {}
    for dl in dlines:
        terms = lf_terms(dl)
        for t in terms:
            if bool(re.match(r'[\w:]+\|',t)):
                counts[t] = counts.get(t,0) + 1
                pos, word_form = t.split('|')
                if pos=='part' and dset_name=='hagar':
                    pos='v'
    csv_lines = [{'word-form':k.split('|')[1], 'pos-marking':k.split('|')[0], 'count':v} for k,v in counts.items()]
    df = pd.DataFrame(csv_lines)
    df = df.sort_values('pos-marking',key=lambda series: series.apply(lambda x:-np.inf if x.startswith('v') else hash(x)))
    df.index = list(range(len(df)))
    df.to_csv(f'{dset_name}_lf_vocab_and_counts.csv')
    return df

def counts_from_conll_lines(dlines,dset_name):
    counts = {}
    for dl in dlines:
        if dl == '\n':
            continue
        row = dl.split('\t')
        wf, lemma, udtag, chiltag = row[1:5]
        lempos = f'{wf}|{lemma}|{udtag}|{chiltag}'
        if chiltag=='part' and dset_name=='hagar':
            chiltag='v'
            if udtag!='VERB':
                print(udtag)
        counts[lempos] = counts.get(lempos,0) + 1
    csv_lines = [{'word-form':k.split('|')[0], 'lemma':k.split('|')[1], 'udtag':k.split('|')[2], 'chiltag':k.split('|')[3], 'count':v} for k,v in counts.items()]
    df = pd.DataFrame(csv_lines)
    df = df.sort_values('udtag',key=lambda series: series.apply(lambda x:-np.inf if x=='VERB' else hash(x)))
    df.index = list(range(len(df)))
    df.to_csv(f'{dset_name}_conll_vocab_and_counts.csv')
    return df

adam_df = counts_from_conll_lines(aconlines, 'adam')
hagar_df = counts_from_conll_lines(hconlines, 'hagar')
print(hagar_df.loc[(hagar_df['udtag']=='VERB') & (hagar_df['chiltag']!='v')])
breakpoint()
