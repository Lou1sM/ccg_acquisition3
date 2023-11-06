import json
import re
from utils import split_respecting_brackets, is_bracketed, outermost_first_chunk, maybe_debrac


def decommafy(parse, debrac=False):
    if len(re.findall(r'[(),]',parse)) == 0:
        return parse
    #maybe_lambda_body = re.search(r'(?<=^lambda \$1_\{(r|e|<r,t>)\}\.).*$',parse)
    maybe_lambda_body = re.search(r'^lambda \$1_\{(r|e|<r,t>)\}\.(.*)$',parse)
    #if parse.startswith('BARE'):
        #breakpoint()
    if maybe_lambda_body is None:
        body = parse
        prefix = ''
    else:
        body = maybe_lambda_body.group(2)
        prefix = parse[:-len(body)]
        assert prefix + body == parse
    suffix = ''
    if body.startswith('Q('):
        body = body[2:-1]
        prefix += 'Q ('
        suffix += ')'
    first_chunk, rest = outermost_first_chunk(body)
    end_of_predicate = first_chunk.find('(')
    pred = first_chunk[:end_of_predicate]
    args = first_chunk[end_of_predicate+1:-1]
    #if pred.startswith('mod|do'):
    if re.match(r'(v|mod)\|do',pred):
        if '-' in pred and args.startswith('v|'):
            marking = pred.split('-')[1].split('_')[0]
            args = re.sub(r'v\|([a-zA-Z0-9]+)_',fr'v|\1-{marking}_', args)
            pred = ''
        elif args.startswith('v|'):
            pred = ''
    arg_splits = split_respecting_brackets(args,sep=',')

    recursed_list = [decommafy(x) for x in arg_splits]
    #recursed = ' '.join(['('+x+')' if len(x.split())>1 and i > 0 else x for i,x in enumerate(recursed_list)])
    recursed = ' '.join(recursed_list)
    converted = maybe_debrac(recursed) if pred=='' else f'{pred} {recursed}'
    #converted = f'{pred} ({recursed})' if 'not' in pred or 'will' in pred else f'{pred} {recursed}'
    #if rest.startswith(','): rest = rest[1:]
    assert rest == ''
    #converted_rest = decommafy(rest)
    #if len(converted_rest) > 0:
    #    breakpoint()
    #    if converted.startswith('and'):
    #        converted = f'({converted}) ({converted_rest})'
    #    else:
    #        converted = f'{converted} ({converted_rest})'
    #assert ' )' not in converted
    assert ',' not in converted or re.search(r'lambda \$\d{1,2}_\{<[ert,<>]+>\}',converted)
    lf = prefix + converted + suffix
    if debrac:
        lf = maybe_debrac(lf)
    elif not is_bracketed(lf) and len(lf.split())>1:
        lf = f'({lf})'
    #else: # already bracced and shouldn't be debracced
    return lf

def lf_preproc(lf_):
    if lf_.startswith('Sem: lambda $2_{<r,t>}.lambda $0_{r}'):
        return None
    lf = lf_.rstrip('\n')
    lf = lf[5:] # have already checked it starts with 'Sem: '
    if lf.startswith('lambda $0'):
        lf = lf.replace('lambda $0_{r}.','').replace('lambda $0_{<r,t>}.','')
        lf = lf.replace(',$0','').replace(',$0','').replace('($0)','')
    dlf = decommafy(lf, debrac=True)
    if ARGS.print_conversions:
        print(f'{lf} --> {dlf}')
    return dlf

def sent_preproc(sent):
    return sent[6:].rstrip('?.!\n').split()

if __name__ == '__main__':
    import argparse
    ARGS = argparse.ArgumentParser()
    ARGS.add_argument("-d", "--dset", type=str, choices=['adam', 'hagar'])
    ARGS.add_argument("-p", "--print_conversions", action='store_true')
    ARGS = ARGS.parse_args()

    with open(f'data/{ARGS.dset}_comma_format.txt') as f:
        adam_lines = f.readlines()

    sents = [adam_lines[i] for i in range(0,len(adam_lines),4)]
    assert all([s.startswith('Sent: ') for s in sents])
    lfs = [adam_lines[i] for i in range(1,len(adam_lines),4)]
    assert all([lf.startswith('Sem: ') for lf in lfs])
    assert all([adam_lines[i]=='example_end\n' for i in range(2,len(adam_lines),4)])
    assert all([adam_lines[i]=='\n' for i in range(3,len(adam_lines),4)])

    if ARGS.dset == 'adam':
        exclude_list = ['don \'t Adam foot', 'who is going to become a spider', 'two Adam', 'a d a m']
    else:
        exclude_list = []
    dset_data = []
    for l,s in zip(lfs,sents):
        if s[6:-3] in exclude_list:
            print(888)
            continue
        pl = lf_preproc(l)
        if pl is None:
            continue
        ps = sent_preproc(s)
        dset_data.append({'lf':pl, 'words':ps})
    dset = {'data':dset_data}

    breakpoint()
    with open(f'data/simplified_{ARGS.dset}.json','w') as f:
        json.dump(dset,f)
