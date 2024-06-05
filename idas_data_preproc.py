import json
from pprint import pprint as pp
import numpy as np
import re
from utils import split_respecting_brackets, is_bracketed, outermost_first_chunk, maybe_debrac, is_wellformed_lf
from converter_config import manual_ida_fixes, he_chars, exclude_lfs, exclude_sents, premanual_ida_fixes, manual_sent_fixes, sent_fixes
from learner_config import pos_marking_dict


with open('data/hagar_comma_format.txt') as f:
    d=f.read()
cw_words = set(sorted(re.findall(fr'(?<=co\|)[{he_chars}]+(?=\()',d)))

maybe_det_str = fr'pro:\w*\|that_\d|qn\|[\w{he_chars}]*|det:\w*\|[\w{he_chars}]*|det\|ha|det\|\~ha|BARE|n:prop\|\w{he_chars}*\'s\''
def maybe_detnoun_match(x):
    return re.match(fr'({maybe_det_str})\((\$\d{{1,2}}),([\w:]+\|[{he_chars}\w-]*)\(\2\)\)',x)

def maybe_attrib_noun_match(x):
    return re.match(fr'({maybe_det_str})\((pro:(sub|obj|per|dem)\|\w*),([\w:]+\|[{he_chars}\w-]*)\(\2\)\)',x)

def is_nplike(x):
    if x in ['WHO', 'WHAT', 'WHOSE', 'you']:
        return True
    if x in ['not', 'and']:
        return False
    elif x.startswith('lambda'):
        return False
    elif bool(maybe_detnoun_match(x)):
        return True
    elif any(c in x for c in ',().$'):
        return False
    else:
        return pos_marking_dict[x.split('|')[0]] == set(['NP'])

def is_adj(x):
    if bool(maybe_detnoun_match(x)):
        return False
    else:
        return x.split('|')[0] in ['adj', 'n']

def decommafy(parse, debrac=False):
    if parse == ARGS.db:
        breakpoint()
    if len(re.findall(r'[(),]',parse)) == 0:
        return parse
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
    return lf

def _decommafy_inner(parse):
    parse = maybe_debrac(parse)
    if ',' not in parse and '(' not in parse:
        assert ')' not in parse
        return parse
    if bool(mdm := maybe_detnoun_match(parse)):
        det, var, noun = mdm.groups()
        noun_pos, noun_word = noun.split('|')
        if noun_pos != 'n':
            print(parse)
        if det == 'BARE':
            return f'n|{noun_word}-BARE'
        else:
            if 'that' in det:
                det = 'det:dem|that'
            assert det.startswith('det') or det.startswith('qn')
        return f'{det} n|{noun_word}'
    if bool(mam := maybe_attrib_noun_match(parse)):
        det, subj, _, noun = mam.groups()
        noun_pos, noun_word = noun.split('|')
        return f'v|equals {subj} ({det} n|{noun_word})'
    else:
        first_chunk, rest = outermost_first_chunk(parse)
        assert rest == ''
        end_of_predicate = first_chunk.find('(')
        pred = first_chunk[:end_of_predicate]
        args = maybe_debrac(first_chunk[end_of_predicate:])
        arg_splits = split_respecting_brackets(args,sep=',')

        if len(arg_splits)==1 and is_nplike(arg_splits[0]):
            if is_nplike(pred):
                if pred.startswith('pro:exist|') or pred in ('WHAT,WHO'):
                    return f'v|equals {decommafy(pred)} {decommafy(arg_splits[0])}'
                else:
                    return f'v|equals {decommafy(arg_splits[0])} {decommafy(pred)}'
            elif is_adj(pred):
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
        if not ( ',' not in lf or re.search(r'lambda \$\d{1,2}_\{<[ert,<>]+>\}',lf)):
            breakpoint()
        return lf

def fix_posses(lf):
    old_lf = lf
    lf = lf.replace('post|', 'adv|')
    if ARGS.dset == 'hagar':
        lf = lf.replace('part|','v|')

    lf = lf.replace('co|like', 'v|like')
    lf = lf.replace('conj|like', 'v|like')
    lf = lf.replace('co|look', 'v|look')
    lf = re.sub(r'BARE\(\$\d{1,2},n\|ʔābaʔ\(\$\d{1,2}\)\)', 'n:prop|ʔābaʔ', lf )
    lf = re.sub(r'BARE\(\$\d{1,2},n\|ʔīmaʔ\(\$\d{1,2}\)\)', 'n:prop|ʔīmaʔ', lf )
    lf = lf.replace('n|ʔābaʔ', 'n:prop|ʔābaʔ' )
    lf = lf.replace('n|ʔīmaʔ', 'n:prop|ʔīmaʔ' )
    lf = re.sub(r'v\|do\b','mod|do',lf) # make 'do' a modal verb
    lf = re.sub(r'_\d{1,2}\b','',lf) # remove sense numbers, way too fine-grained
    lf = re.sub(r'BARE\(\$\d{1,2},(pro:indef\|[{he_chars}\w-]+)\(\$\d{1,2}\)\)', r'\1',lf)
    lf = re.sub(r'BARE\(\$\d{1,2},(det:num|n:let|on|co)\|([{he_chars}\w-]+)\(\$\d{1,2}\)\)', r'n:prop|\2',lf)
    lf = re.sub(r'BARE\(\$\d{1,2},(part\|[{he_chars}\w-]+)\(\$\d{1,2}\)\)', r'\1',lf)
    if lf != old_lf:
        assert 'that one' not in ' '.join(sent)
    lf = re.sub(r'co\|([\w{he_chars}]+\()(?!\$\d|$)',r'v|\1',lf) #'(' inside group to not replace single term
    lf =re.sub(rf',co\|[\w{he_chars}]+', '', lf)
    lf =re.sub(rf'co\|[\w{he_chars}]+,', '', lf)
    lf =re.sub(rf'co\|[\w{he_chars}]+', '', lf)
    lf = lf.replace('()', '')
    return lf

def lf_preproc(lf, sent):
    """Conversion of LFs that is not related to removing commas or fixing det-nouns."""
    #lf = lf_.rstrip('\n')
    #lf = lf[5:] # have already checked it starts with 'Sem: '
    if np_marked:=lf.startswith('BARE($0'):
        lf=lf[8:-1]
        lf = lf.replace(',$0','').replace(',$0','').replace('($0)','')
    if lf in premanual_ida_fixes:
        old_lf = lf
        lf = premanual_ida_fixes[lf]
        if ARGS.print_fixes:
            print(f'Fixing: {old_lf} --> {lf}')

    assert not ('n:prop|ʔābaʔ' in lf or 'n:prop|ʔīmaʔ' in lf) or not ('det|ha' in lf or 'det|~ha' in lf)
    lf = fix_posses(lf)
    if 'lambda $0' in lf.rpartition('.')[0]:
        lf = lf.replace('lambda $0_{r}.','').replace('lambda $0_{<r,t>}.','')
        lf = lf.replace(',$0','').replace(',$0','').replace('($0)','')
    maybe_wh_lambda_match = re.match(r'^lambda (\$\d)_\{(r|e|<<e,e>,e>)}\.',lf)
    if is_wh:=(bool(maybe_wh_lambda_match)):
        wh_var_with_num = maybe_wh_lambda_match.groups()[0]
        wh_word = sent.split()[0]
        if ARGS.dset == 'hagar':
            replacer = 'WH'
        elif (wh_word in ['what','who']):
            replacer = wh_word.upper()
        elif wh_word == 'which':
            replacer = 'det:dem|WHICH'
        elif 'what' in sent: # for in-situs
            replacer = 'WHAT'
        elif 'who' in sent: # for in-situs
            replacer = 'WHO'
        elif 'which' in sent: # for in-situs
            replacer = 'det:dem|WHICH'
        elif wh_word not in ['why', 'how']:
            breakpoint()
        if wh_word not in ['why', 'how']:
            matched = maybe_wh_lambda_match.group()
            lf = lf.replace(matched,'')
            lf = 'Q(' + lf.replace(wh_var_with_num,f'pro:int|{replacer}') + ')'

    if re.search(r'(?<![a-zA-Z])v\|(?!do)', lf):
        lf = re.sub(r'(?<![a-zA-Z])v\|do(?![a-zA-Z])','mod|do',lf)

    if '(pro:rel|that)' in lf or ',pro:rel|that)' in lf or '(pro:rel|that,' in lf or 'pro:rel|that,n|' in lf:
        lf = lf.replace('pro:rel|that','pro:dem|that')
    if re.search(r'pro:rel\|that\(\$\d,(n|pro:indef)', lf):
        lf = lf.replace('pro:rel|that','det:dem|that')
    if lf in ['Q', 'and', '(you)', '(WHO)', 'aux|~be((WHAT))']:
        return ''
    dlf = decommafy(lf, debrac=True)
    if ARGS.print_conversions:
        print(f'{lf} --> {dlf}', end='\t')
    if dlf in manual_ida_fixes.keys():
        old_dlf = dlf
        dlf = manual_ida_fixes[dlf]
        if ARGS.print_fixes:
            print(f'Fixing: {old_dlf} --> {dlf}')
    if dpoint['idasent'].lstrip('Adam ').startswith('is that') and not dlf.startswith('Q '):
        dlf = f'Q ({dlf})'
    if dlf == 'v|equals pro:dem|that (det:art|a n|racket)':
        breakpoint()
    dlf = re.sub(r'BARE \$\d{1,2} \((det:num\|[{he_chars}\w-]+) \((n\|[\w{he_chars}-]+) \$\d{1,2}\)\)', r'\1 \2', dlf)
    assert is_wellformed_lf(dlf)
    return dlf

def sent_preproc(lf, sent):
    #sent = sent[6:].rstrip('?.!\n ')
    sent = sent.rstrip('?.! ')
    assert not sent.endswith(' ')
    if sent in sent_fixes:
        sent = sent_fixes[sent]
    sent = [w for w in sent.split() if f'co|{w}' not in lf and not(w=='we' and w not in lf)]
    if len(sent)>0 and sent[0] == 'ʔavāl' and 'ʔavāl' not in lf:
        sent = sent[1:]
    if sent == '':
        breakpoint()
    sent = [w for w in sent if not w[0].isupper() or w.lower() in lf]
    if ' '.join(sent) in manual_sent_fixes:
        sent = manual_sent_fixes[' '.join(sent)].split()
    return sent

def decide_if_question(lf, sent, udtags):
    conjs = ('but', 'and', 'or')
    wh_words = ('what', 'who', 'how', 'where', 'when', 'which')
    if udtags[0] == 'CONJ' and sent[0] not in conjs:
        breakpoint()
    for conj in conjs:
        if sent[0] == conj:
            sent = sent[1:]
            if lf.startswith(conj):
                assert lf.starswith(f'{conj} (') and lf.endswith(')')
                lf = lf[len(conj)+2:-1]
    if 'you' not in lf.split() and udtags[0] in ('AUX', 'VERB'):
        if not ( lf.startswith('Q ')):
            print('should have Q in lf, adding one:', sent, lf)
            lf = f'Q ({lf})'
    #elif any(w in wh_words for w in sent):
    elif sent[0] in wh_words:
        if not ( lf.startswith('Q ')):
            print('probably should have Q in lf:', sent, lf)
    elif lf.startswith('Q '):
        print('marked as Q without leading verb, removing:', sent, lf)
        lf = lf[3:-1]
    return lf, sent

if __name__ == '__main__':
    import argparse
    ARGS = argparse.ArgumentParser()
    ARGS.add_argument("-d", "--dset", type=str, choices=['adam', 'hagar'], default='adam')
    ARGS.add_argument("--db", type=str)
    ARGS.add_argument("--db_sent", type=str)
    ARGS.add_argument("-p", "--print_conversions", action='store_true')
    ARGS.add_argument("-f", "--print_fixes", action='store_true')
    ARGS = ARGS.parse_args()

    #with open(f'data/{ARGS.dset}_comma_format.txt') as f:
        #dset_lines = f.readlines()

    with open(f'data/{ARGS.dset}-combined-info.json') as f:
        dset = json.load(f)
    #sents = [dset_lines[i] for i in range(0,len(dset_lines),4)]
    #assert all([s.startswith('Sent: ') for s in sents])
    #lfs = [dset_lines[i] for i in range(1,len(dset_lines),4)]
    #assert all([lf.startswith('Sem: ') for lf in lfs])
    #assert all([dset_lines[i]=='example_end\n' for i in range(2,len(dset_lines),4)])
    #assert all([dset_lines[i]=='\n' for i in range(3,len(dset_lines),4)])

    dset_data = []
    n_excluded = 0
    with open('../CHILDES_UD2LF_2/conll/full_adam/adam.all.udv1.conllu.final') as f:
        conll = f.read().strip().split('\n\n')

    with open('../CHILDES_UD2LF_2/LF_files/full_adam/adam.all_lf.txt') as f:
        ida_lf = f.read().strip().split('\n\n')

    #for l,s in zip(lfs,sents):
    for dpoint in dset:
        sent, lf, udtags = dpoint['idasent'], dpoint['idalf'], dpoint['udtags']
        if any(x in dpoint['idalf'] for x in exclude_lfs):
            n_excluded+=1
            continue
        #if s[6:-3] in exclude_sents:
        if sent[:-2] in exclude_sents:
            n_excluded+=1
            continue
        ps = sent_preproc(lf, sent)
        if len(ps) > 0 and ps[0] == 'ʔavāl':
            breakpoint()
        if ARGS.db_sent is not None and ps == ARGS.db_sent.split():
            breakpoint()
        if ps == []:
            n_excluded+=1
            continue
        pl = lf_preproc(lf, sent)
        if pl.endswith(': ') or pl is None or '_' in pl or pl=='':
            n_excluded+=1
            continue
        if pl=="not (n:prop|adam's $0 (n|mouth $0))":
            breakpoint()
        if not all(f'lambda {v}.' in pl for v in set(re.findall(r'\$\d{1,2}', pl))):
            continue
        assert not any(x in pl for x in exclude_lfs)
        pl, ps = decide_if_question(pl, ps, udtags)
        if ARGS.print_conversions:
            print(' '.join(ps))
        if pl == 'Q ()':
            breakpoint()
        dset_data.append({'lf':pl, 'words':ps})
    x=[' '.join(y['words']) for y in dset_data]
    y=dict(zip(*np.unique(x, return_counts=True)))
    hist_counts = sorted([(k,int(v)) for k,v in y.items()], key=lambda x:x[1], reverse=True)
    pp(hist_counts[:20])
    dset = {'data':dset_data, 'hist_counts':hist_counts}

    print(f'Num Excluded Points: {n_excluded}')
    print(f'Num Remaining Points: {len(dset_data)}')
    with open(f'data/{ARGS.dset}.json','w') as f:
        json.dump(dset,f)
    with open(f'data/{ARGS.dset}_no_commas.txt','w') as f:
        for dpoint in dset_data:
            sent, lf = ' '.join(dpoint['words']), dpoint['lf']
            f.write(f'Sent: {sent}\n')
            f.write(f'Sem: {lf}\n\n')
