import json
import re
from utils import split_respecting_brackets, is_bracketed, outermost_first_bracketed_chunk, maybe_debrac


def decommafy(parse):
    #if ',_' in parse or '_,' in parse:
        #return None
    if len(re.findall(r'[(),]',parse)) == 0:
        return parse
    maybe_lambda_body = re.search(r'(?<=^lambda \$1_\{[re]\}\.).*$',parse)
    if maybe_lambda_body is None:
        body = parse
        prefix = ''
    else:
        body = maybe_lambda_body.group()
        prefix = parse[:maybe_lambda_body.span()[0]]
        assert prefix + body == parse
    suffix = ''
    if body.startswith('Q('):
        body = body[2:-1]
        prefix += 'Q ('
        suffix += ')'
    first_chunk, rest = outermost_first_bracketed_chunk(body)
    end_of_predicate = first_chunk.find('(')
    pred = first_chunk[:end_of_predicate]
    arg_splits = split_respecting_brackets(first_chunk[end_of_predicate+1:-1],sep=',')
    recursed_list = [decommafy(x) for x in arg_splits]
    recursed = ' '.join(['('+x+')' if len(x.split())>1 and i > 0 else x for i,x in enumerate(recursed_list)])
    #converted = f'{pred} {recursed}'
    converted = f'{pred} ({recursed})' if 'not' in pred or 'will' in pred else f'{pred} {recursed}'
    if rest.startswith(','): rest = rest[1:]
    assert rest == ''
    #converted_rest = decommafy(rest)
    #if len(converted_rest) > 0:
    #    breakpoint()
    #    if converted.startswith('AND'):
    #        converted = f'({converted}) ({converted_rest})'
    #        print(converted)
    #    else:
    #        converted = f'{converted} ({converted_rest})'
    #assert ' )' not in converted
    assert ',' not in converted or re.search(r'lambda \$\d{1,2}_\{<[ert,<>]+>\}',converted)
    print(f'{parse} --> {converted}')
    lf = prefix + converted + suffix
    return lf

def lf_preproc(lf_):
    if lf_.startswith('Sem: lambda $2_{<r,t>}.lambda $0_{r}'):
        return None
    lf = lf_.rstrip('\n')
    lf = lf[5:] # have already checked it starts with 'Sem: '
    lf = lf.replace('lambda $0_{r}.','').replace(',$0','')
    lf = lf.replace('lambda $0_{<r,t>}.','').replace(',$0','')
    return decommafy(lf)
    #m = re.search(r'(?<=^Sem: lambda \$0_\{r\}\.)(.*)(,\$0\))(\)*$)',lf)
    #body, _, closing_brackets = m.groups()
    #return body+closing_brackets

def sent_preproc(sent):
    return sent[6:].rstrip('?.!\n').split()

with open('data/adam.all_lf.txt') as f:
    adam_lines = f.readlines()

sents = [adam_lines[i] for i in range(0,len(adam_lines),4)]
assert all([s.startswith('Sent: ') for s in sents])
lfs = [adam_lines[i] for i in range(1,len(adam_lines),4)]
assert all([lf.startswith('Sem: ') for lf in lfs])
assert all([adam_lines[i]=='example_end\n' for i in range(2,len(adam_lines),4)])
assert all([adam_lines[i]=='\n' for i in range(3,len(adam_lines),4)])

dset_data = []
for l,s in zip(lfs,sents):
    pl = lf_preproc(l)
    if pl is None:
        continue
    ps = sent_preproc(s)
    dset_data.append({'lf':pl, 'words':ps})
dset = {'data':dset_data}

breakpoint()
with open('data/simplified_adam.json','w') as f:
    json.dump(dset,f)
