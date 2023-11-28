import json
from pprint import pprint as pp
import numpy as np
import re
from utils import split_respecting_brackets, is_bracketed, outermost_first_chunk, maybe_debrac
from config import manual_ida_fixes, pos_marking_dict


def maybe_detnoun_match(x):
    return re.match(r'(pro:\w*\|that_\d|qn\|\w*|det:\w*\|\w*|BARE|n:prop\|\w*\'s\')\((\$\d{1,2}),(\w*\|[a-z0-9_-]*)\(\2\)\)',x)

def maybe_attrib_noun_match(x):
    return re.match(r'(qn\|\w*|det:\w*\|\w*|BARE|n:prop\|\w*\'s\')\((pro:(sub|obj|per|dem)\|\w*),(\w*\|[a-z0-9_-]*)\(\2\)\)',x)

def is_nplike(x):
    if x in ['WHO', 'WHAT', 'you']:
        return True
    if x == 'not':
        return False
    elif x.startswith('lambda'):
        return False
    elif bool(maybe_detnoun_match(x)):
        return True
    elif any(c in x for c in ',().\$'):
        return False
    else:
        return pos_marking_dict[x.split('|')[0]] == set(['NP'])

def is_adj(x):
    if bool(maybe_detnoun_match(x)):
        return False
    else:
        #return x.split('|')[0] in ['part', 'adj']
        return x.split('|')[0] in ['adj', 'n']

def decommafy(parse, debrac=False):
    if ARGS.dset == 'hagar':
        parse = parse.replace('part|','v|')
    if parse == ARGS.db:
        breakpoint()
    if len(re.findall(r'[(),]',parse)) == 0:
        return parse
    #maybe_lambda_body = re.search(r'(?<=^lambda \$1_\{(r|e|<r,t>)\}\.).*$',parse)
    maybe_lambda_body = re.match(r'^lambda (WHAT|WHO|\$1_\{(r|e|<r,t>)\})\.(.*)$',parse)
    if maybe_lambda_body is None:
        body = parse
        prefix = ''
    else:
        body = maybe_lambda_body.group(3)
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
    #lf = lf.replace('you_2','you_1').replace('you_3','you_1')
    lf = re.sub(r'_\d{1,2}\b','',lf)
    return lf

def _decommafy_inner(parse):
    if parse == 'pro:dem|that_1(pro:per|it_3)':
        return 'equals pro:dem|that_1 pro:per|it_3'
    if parse.startswith('Q(mod|do_1('):
        breakpoint()
    if bool(mdm := maybe_detnoun_match(parse)):
        det, var, noun = mdm.groups()
        if det != 'BARE':
            if 'that' in det:
                #print(det)
                det = f'det:dem|that_1'
            assert det.startswith('det') or det.startswith('qn')
        noun_pos, noun_word = noun.split('|')
        #if noun_pos != 'n':
            #print(noun_pos)
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
            elif args.startswith('v|'): # removing 'do'
                pred = ''
        arg_splits = split_respecting_brackets(args,sep=',')

        #if parse == 'Q(n|right_3(pro:dem|that_1))':
            #breakpoint()
        if len(arg_splits)==1 and is_nplike(arg_splits[0]):
            if is_nplike(pred):
                if pred.startswith('pro:exist|') or pred in ('WHAT,WHO'):
                    return f'equals {decommafy(pred)} {decommafy(arg_splits[0])}'
                else:
                    return f'equals {decommafy(arg_splits[0])} {decommafy(pred)}'
            elif is_adj(pred):
                #if pred.startswith('n|'):
                    #print(f'hasproperty {decommafy(arg_splits[0])} {decommafy(pred)}')
                dpred = decommafy(pred)
                dpred_pos, dpred_word = dpred.split('|')
                return f'hasproperty {decommafy(arg_splits[0])} adj|{dpred_word}'
        recursed_list = [decommafy(x) for x in arg_splits]
        if pred in ['and', '']: # can be '' because do-support
            debracced_rl = [maybe_debrac(recursed_list[0])] + recursed_list[1:]
            lf = ' '.join(debracced_rl)
        else:
            recursed = ' '.join(recursed_list)
            lf = f'{pred} {recursed}'
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

def lf_preproc(lf_, sent):
    lf = lf_.rstrip('\n')
    lf = lf[5:] # have already checked it starts with 'Sem: '
    #if lf == 'lambda $0_{r}.pro:poss|mine_3(pro:per|it_1,$0)':
    #    breakpoint()
    if 'lambda $0' in lf.rpartition('.')[0]:
        lf = lf.replace('lambda $0_{r}.','').replace('lambda $0_{<r,t>}.','')
        lf = lf.replace(',$0','').replace(',$0','').replace('($0)','')
    if lf.startswith('lambda $1_{e}'):
        if (wh_word := sent.split()[0] in ['what','who']):
            lf = lf[14:].replace('$1',f'{wh_word.upper()}1')
        elif 'what' in sent:
            lf = lf[14:].replace('$1',f'WHAT')
        elif 'who' in sent:
            lf = lf[14:].replace('$1',f'WHO')
    dlf = decommafy(lf, debrac=True)
    if ARGS.print_conversions:
        print(f'{lf} --> {dlf}')
    if dlf in manual_ida_fixes.keys():
        old_dlf = dlf
        dlf = manual_ida_fixes[dlf]
        print(f'fixing {old_dlf} to {dlf}')
    return dlf

def sent_preproc(sent):
    return sent[6:].rstrip('?.!\n').split()

if __name__ == '__main__':
    import argparse
    ARGS = argparse.ArgumentParser()
    ARGS.add_argument("-d", "--dset", type=str, choices=['adam', 'hagar'], required=True)
    ARGS.add_argument("--db", type=str)
    ARGS.add_argument("-p", "--print_conversions", action='store_true')
    ARGS = ARGS.parse_args()

    with open(f'data/{ARGS.dset}_comma_format.txt') as f:
        dset_lines = f.readlines()

    sents = [dset_lines[i] for i in range(0,len(dset_lines),4)]
    assert all([s.startswith('Sent: ') for s in sents])
    lfs = [dset_lines[i] for i in range(1,len(dset_lines),4)]
    assert all([lf.startswith('Sem: ') for lf in lfs])
    assert all([dset_lines[i]=='example_end\n' for i in range(2,len(dset_lines),4)])
    assert all([dset_lines[i]=='\n' for i in range(3,len(dset_lines),4)])

    if ARGS.dset == 'adam':
        exclude_list = ['don \'t Adam foot', 'who is going to become a spider', 'two Adam', 'a d a m']
    else:
        exclude_list = []
    dset_data = []
    for l,s in zip(lfs,sents):
        if s[6:-3] in exclude_list:
            continue
        pl = lf_preproc(l,s)
        if pl is None:
            continue
        ps = sent_preproc(s)
        dset_data.append({'lf':pl, 'words':ps})
    x=[' '.join(y['words']) for y in dset_data]
    y=dict(zip(*np.unique(x, return_counts=True)))
    hist_counts = sorted([(k,int(v)) for k,v in y.items()], key=lambda x:x[1], reverse=True)
    pp(hist_counts[:20])
    dset = {'data':dset_data, 'hist_counts':hist_counts}

    with open(f'data/{ARGS.dset}.json','w') as f:
        json.dump(dset,f)
