import re
from utils import split_respecting_brackets, is_bracketed, outermost_first_bracketed_chunk
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
    if words[-1] == "'.'":
        print(words)
        words = words[:-1]
        print(words)
        print()

    assert "'.'" not in words
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
    if parse_str.startswith('('):
        breakpoint()

    return words, parse_str

def convert_to_no_comma_form(parse):
    if len(re.findall(r'[(),]',parse)) == 0:
        return parse
    end_of_lambda = parse.rfind('.')
    to_split = parse[end_of_lambda+1:] if end_of_lambda!=-1 else parse
    first_chunk, rest = outermost_first_bracketed_chunk(to_split)
    if first_chunk.startswith('('):
        if '(' in first_chunk[1:]: # not just list of variables
            arg_splits = split_respecting_brackets(first_chunk[1:-1],sep=',')
            recursed = ' '.join(sorted(['('+convert_to_no_comma_form(x)+')' for x in arg_splits]))
            converted = f'AND {recursed}'
        else:
            converted = convert_to_no_comma_form(to_split[1:-1])
    else:
        end_of_predicate = first_chunk.find('(')
        pred = first_chunk[:end_of_predicate]
        arg_splits = split_respecting_brackets(first_chunk[end_of_predicate+1:-1],sep=',')
        recursed = ' '.join([convert_to_no_comma_form(x) for x in arg_splits])
        converted = f'{pred} {recursed}'
    if rest.startswith(','): rest = rest[1:]
    converted_rest = convert_to_no_comma_form(rest)
    if len(converted_rest) > 0:
        converted = f'{converted} ({converted_rest})'
    assert ' )' not in converted
    assert ',' not in converted
    print(f'{parse} --> {converted}')
    if end_of_lambda == -1:
        return converted
    else:
        return parse[:end_of_lambda+1] + converted


dpoints = []
for gl in geoquery_data:
    words, parse = process_line(gl)
    print(f'\n{parse}\n')
    decommaified = convert_to_no_comma_form(parse)
    print(parse,'\t',decommaified)
    dpoints.append({'words':words,'parse':decommaified,'parse_with_commas':parse})

np_list = list(set(np_list))
print(np_list)
print(intransitives)
print(transitives)
processed_dset = {'np_list':np_list, 'intransitive_verbs':list(intransitives),
                    'transitive_verbs': list(transitives), 'data':dpoints}
with open('data/preprocessed_geoqueries.json','w') as f:
    json.dump(processed_dset,f)
