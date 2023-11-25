import subprocess as sp
import pandas as pd
import sys


df=pd.read_csv('hebrew_latin_table.tsv', sep='\t')
alphabet_trans = dict(zip(df['Unnamed: 0'], df['Unnamed: 1']))
alphabet_trans = {k:'' if v=='vowel' or 'stress' in v else v for k,v in alphabet_trans.items() if isinstance(v,str)}
alphabet_trans['̌'] = ''
assert all(isinstance(x,str) for x in alphabet_trans.values())

he_latin = sys.argv[1]
he = he_latin
for k,v in alphabet_trans.items():
    he = he.replace(k,v)
print(f'Hebrew script: {he}\n')
print('English translation:')
sp.call(['/usr/bin/trans','he:en',he])
