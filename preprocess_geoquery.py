import re
import json

with open('geoqueries880') as f:
    geoquery_data = f.readlines()

#var_name_conversion_dict = {'A':'x','B':'y','C':'z','D':'u','E':'v','F':'s','G':'t'}
var_name_conversion_dict = {k:f'${i}' for i,k in enumerate('ABCDEFG')}
intransitives = set([x[0] for x in re.findall(r'(\w+)(?=(\([A-G]\)))',''.join(geoquery_data))])
transitives = set([x[0] for x in re.findall(r'(\w+)(?=(\([A-G],[A-G]\)))',''.join(geoquery_data))])
np_list = []

def process_line(g_line):
    assert g_line.startswith('parse(')
    g_line = g_line.lstrip('parse(').rstrip('.\n')[:-1]
    words_str,_,parse_str = g_line[:-1].partition(', answer(A,')
    words = list(words_str[1:-1].split(','))
    if '\+' in parse_str:
        print(parse_str)
    parse_str = re.sub(r'\\\+','not',parse_str)
    for vn_old,vn_new in var_name_conversion_dict.items():
        parse_str = re.sub(vn_old,vn_new,parse_str)
    # replace 'const' predicate with constants as names
    consts = re.findall(r'const\(\$\d,\w+id\(\'?[a-z ,]+\'?[,_]*\)\)',parse_str)
    for c in consts:
        parse_str = re.sub(re.escape(','+c),'',parse_str)
        parse_str = re.sub(re.escape(c+','),'',parse_str)
        assert c[6:8] in var_name_conversion_dict.values()
        var_str_to_match = fr'\${c[7]}'
        name_being_given = c.split('id(')[1][:-2]
        np_list.append(name_being_given)
        parse_str = re.sub(var_str_to_match,name_being_given,parse_str)
    parse_str = 'lambda $0.' + parse_str
    if 'const' in parse_str:
        print(parse_str)
    return words, parse_str

np_list = list(set(np_list))
dpoints = []
for gl in geoquery_data:
    words, parse = process_line(gl)
    dpoints.append({'words':words,'parse':parse})

processed_dset = {'np_list':np_list, 'intransitive_verbs':list(intransitives),
                    'transitive_verbs': list(transitives), 'data':dpoints}
with open('preprocessed_geoqueries.json','w') as f:
    json.dump(processed_dset,f)
