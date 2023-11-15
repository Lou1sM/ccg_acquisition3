import json
from pprint import pprint as pp
import numpy as np
import re
from utils import split_respecting_brackets, is_bracketed, outermost_first_chunk, maybe_debrac


def maybe_detnoun_match(x):
    return re.match(r'(qn\|\w*|det:\w*\|\w*|BARE)\((\$\d{1,2}),(\w*\|\w*)\(\2\)\)',x)

def maybe_attrib_noun_match(x):
    return re.match(r'(qn\|\w*|det:\w*\|\w*|BARE)\((pro:(sub|obj|per|dem)\|\w*),(\w*\|\w*)\(\2\)\)',x)

def is_np(x):
    if x.startswith('lambda'):
        return False
    elif bool(maybe_detnoun_match(x)):
        return True
    elif any(c in x for c in ',().\$'):
        return False
    else:
        return x.split('|')[0] in ['pro:per', 'n:prop', 'pro:dem', 'pro:sub', 'pro:obj', 'pro:exist']

def is_adj(x):
    #if x == 'n|right_3':
        #return True
    if bool(maybe_detnoun_match(x)):
        return False
    else:
        #return x.split('|')[0] in ['part', 'adj']
        return x.split('|')[0] in ['adj', 'n']

def decommafy(parse, debrac=False):
    if len(re.findall(r'[(),]',parse)) == 0:
        return parse
    if parse == 'det:poss|my_3(pro:dem|that_1,n|umbrella_4(pro:dem|that_1))':
        breakpoint()
    #maybe_lambda_body = re.search(r'(?<=^lambda \$1_\{(r|e|<r,t>)\}\.).*$',parse)
    maybe_lambda_body = re.search(r'^lambda \$1_\{(r|e|<r,t>)\}\.(.*)$',parse)
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
    inner_lf = _decommafy_inner(body)
    lf = prefix + inner_lf + suffix
    if debrac:
        lf = maybe_debrac(lf)
    elif not is_bracketed(lf) and len(lf.split())>1:
        lf = f'({lf})'
    return lf

def _decommafy_inner(parse):
    if parse == 'pro:dem|that_1(pro:per|it_3)':
        return 'equals pro:dem|that_1 pro:per|it_3'
    #if 'n|spray_3' in parse:
        #breakpoint()
    if bool(mdm := maybe_detnoun_match(parse)):
        det, var, noun = mdm.groups()
        if det == 'BARE':
            det_word = 'BARE'
        else:
            det_pos, det_word = det.split('|')
            if not det_pos.startswith('det'):
                print(det_pos)
        noun_pos, noun_word = noun.split('|')
        if noun_pos != 'n':
            print(noun_pos)
        return f'{det} n|{noun_word}'
    if bool(mam := maybe_attrib_noun_match(parse)):
        det, subj, _, noun = mam.groups()
        noun_pos, noun_word = noun.split('|')
        #if noun_pos != 'n':
            #print(noun_pos)
        #return f'equals {subj} {det}(lambda $6.{noun} $6)'# 5 highest that's currently in dset
        return f'equals {subj} ({det} n|{noun_word})'
    else:
        first_chunk, rest = outermost_first_chunk(parse)
        assert rest == ''
        end_of_predicate = first_chunk.find('(')
        pred = first_chunk[:end_of_predicate]
        args = first_chunk[end_of_predicate+1:-1]
        if re.match(r'(v|mod)\|do',pred):
            if '-' in pred and args.startswith('v|'):
                marking = pred.split('-')[1].split('_')[0]
                args = re.sub(r'v\|([a-zA-Z0-9]+)_',fr'v|\1-{marking}_', args)
                pred = ''
            elif args.startswith('v|'):
                pred = ''
        arg_splits = split_respecting_brackets(args,sep=',')

        #if parse == 'Q(n|right_3(pro:dem|that_1))':
            #breakpoint()
        if len(arg_splits)==1 and is_np(arg_splits[0]):
            if is_np(pred):
                if pred.startswith('pro:exist|'):
                    return f'equals {decommafy(pred)} {decommafy(arg_splits[0])}'
                else:
                    return f'equals {decommafy(arg_splits[0])} {decommafy(pred)}'
            elif is_adj(pred):
                if pred.startswith('n|'):
                    print(f'has_property {decommafy(arg_splits[0])} {decommafy(pred)}')
                return f'has_property {decommafy(arg_splits[0])} {decommafy(pred)}'
        recursed_list = [decommafy(x) for x in arg_splits]
        recursed = ' '.join(recursed_list)
        lf = maybe_debrac(recursed) if pred=='' else f'{pred} {recursed}'
        #converted_rest = decommafy(rest)
        #if len(converted_rest) > 0:
        #    breakpoint()
        #    if converted.startswith('and'):
        #        converted = f'({converted}) ({converted_rest})'
        #    else:
        #        converted = f'{converted} ({converted_rest})'
        #assert ' )' not in converted
        if not ( ',' not in lf or re.search(r'lambda \$\d{1,2}_\{<[ert,<>]+>\}',lf)):
            breakpoint()
        return lf

def lf_preproc(lf_):
    lf = lf_.rstrip('\n')
    lf = lf[5:] # have already checked it starts with 'Sem: '
    if 'lambda $0' in lf.rpartition('.')[0]:
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
    ARGS.add_argument("-d", "--dset", type=str, choices=['adam', 'hagar'], required=True)
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
            continue
        pl = lf_preproc(l)
        if pl is None:
            continue
        ps = sent_preproc(s)
        dset_data.append({'lf':pl, 'words':ps})
    x=[' '.join(y['words']) for y in dset_data]
    y=dict(zip(*np.unique(x, return_counts=True)))
    hist_counts = sorted([(k,int(v)) for k,v in y.items()], key=lambda x:x[1], reverse=True)
    pp(hist_counts[:20])
    dset = {'data':dset_data, 'hist_counts':hist_counts}

    with open(f'data/simplified_{ARGS.dset}.json','w') as f:
        json.dump(dset,f)
