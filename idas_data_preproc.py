import json
from pprint import pprint as pp
import numpy as np
import re
from utils import split_respecting_brackets, is_bracketed, outermost_first_chunk, maybe_debrac, IWFF, new_var_num, alpha_normalize, all_lambda_body_splits, de_q, add_q, add_not, add_whq
from converter_config import manual_ida_fixes, he_chars, exclude_lfs, exclude_sents, premanual_ida_fixes, manual_sent_fixes, sent_fixes, neg_conts, direct_take_lf_from_sents
from learner_config import pos_marking_dict
from manual_which_dict import manual_which_dict


with open('data/hagar_comma_format.txt') as f:
    d=f.read()
cw_words = set(sorted(re.findall(fr'(?<=co\|)[{he_chars}]+(?=\()',d)))

ings = []
falseings = []

iwff = IWFF()

maybe_det_str = fr'pro:\w*\|that_\d|qn\|[\w{he_chars}]*|det:\w*\|[\w{he_chars}]*|det\|ha|det\|\~ha|BARE|n:prop\|[\w{he_chars}]*\'s'
#maybe_det_str = fr'pro:\w*\|that_\d|qn\|[\w{he_chars}]*|det:\w*\|[\w{he_chars}]*|det\|ha|det\|\~ha|BARE|n:prop\|\w*\'s'
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
    splits = split_respecting_brackets(inner_lf)
    if inner_lf.startswith('v|equals'):
        print(inner_lf)
    if len(splits)==3 and not inner_lf.startswith('v|equals'):
        inner_lf = f'{splits[0]} {splits[2]} {splits[1]}'
    if len(splits)==4 and inner_lf.startswith('v|'):
        inner_lf = f'{splits[0]} {splits[3]} {splits[2]} {splits[1]}'
        print(inner_lf)
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
        #if noun_pos != 'n':
            #print('sth that isnt a noun in noun-like position:', parse)
        if det == 'BARE':
            return f'n|{noun_word}-BARE'
        else:
            if 'that' in det:
                det = 'det:dem|that'
            assert det.startswith('det') or det.startswith('qn') or det.startswith('n:prop')
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
                return f'hasproperty {decommafy(arg_splits[0])} adj|{dpred_word}' # will be inverted
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

def fix_posses(lf, sent):
    old_lf = lf
    lf = lf.replace('post|', 'adv|')
    if ARGS.dset == 'hagar':
        lf = lf.replace('part|','v|')

    lf = lf.replace('co|like', 'v|like')
    lf = lf.replace('conj|like', 'v|like')
    lf = lf.replace('co|look', 'v|look')
    if 'alright' in sent:
        lf = lf.replace('co|alright', 'adj|alright')
    lf = lf.replace('v|ready', 'adj|ready')
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
    lf = lf.replace('mod:aux|', 'mod|')
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
    lf = fix_posses(lf, sent)
    if 'lambda $0' in lf.rpartition('.')[0]:
        lf = lf.replace('lambda $0_{r}.','').replace('lambda $0_{<r,t>}.','').replace('lambda $0_{<<e,e>,e>}.','')
        lf = lf.replace(',$0','').replace(',$0','').replace('($0)','')
    maybe_wh_lambda_match = re.match(r'^lambda (\$\d)_\{(r|e|<<e,e>,e>)}\.',lf)
    if is_wh:=(bool(maybe_wh_lambda_match)):
        wh_var_with_num = maybe_wh_lambda_match.groups()[0]
        wh_word = sent.split()[0]
        if ARGS.dset == 'hagar':
            replacer_word = 'WH'
        elif (wh_word in ['what','who']):
            replacer_word = wh_word.upper()
        elif wh_word == 'which':
            replacer_word = 'WHICH'
        elif 'what' in sent: # for in-situs
            replacer_word = 'WHAT'
        elif 'who' in sent: # for in-situs
            replacer_word = 'WHO'
        elif 'which' in sent: # for in-situs
            replacer_word = 'WHICH'
        elif wh_word not in ['why', 'how']:
            breakpoint()
        if wh_word not in ['why', 'how']:
            matched = maybe_wh_lambda_match.group()
            lf = lf.replace(matched,'')
            lf = 'Q(' + lf.replace(wh_var_with_num,f'pro:int|{replacer_word}') + ')'
            #lf = 'Q(' + lf.replace(wh_var_with_num, replacer_word) + ')'

    if re.search(r'(?<![a-zA-Z])v\|(?!do)', lf):
        lf = re.sub(r'(?<![a-zA-Z])v\|do(?![a-zA-Z])','mod|do',lf)

    if '(pro:rel|that)' in lf or ',pro:rel|that)' in lf or '(pro:rel|that,' in lf or 'pro:rel|that,n|' in lf:
        lf = lf.replace('pro:rel|that','pro:dem|that')
    if re.search(r'pro:rel\|that\(\$\d,(n|pro:indef)', lf):
        lf = lf.replace('pro:rel|that','det:dem|that')
    assert lf not in ['Q', 'and', '(you)', '(WHO)', 'aux|~be((WHAT))']

    if 'tired' in sent:
        lf = re.sub(r'(v|part)\|tire(\-(pastp?))?', 'adj|tired', lf)
        print(sent, lf)
        #lf.replace('v|tire-past', 'adj|tired').replace('part|tire-pastp', 'adj|tired')
    dlf = decommafy(lf, debrac=True)

    if 'was' in sent or 'were' in sent or '\'re' in sent:
        dlf = dlf.replace('hasproperty', 'hasproperty-past')
        dlf = dlf.replace('equals', 'equals-past')
    if ARGS.print_conversions:
        print(f'{lf} --> {dlf}', end='\t')
    #if dpoint['idasent'].lstrip('Adam ').startswith('is that') and not dlf.startswith('Q '):
        #dlf = f'Q ({dlf})'
    dlf = re.sub(r'BARE \$\d{1,2} \((det:num\|[{he_chars}\w-]+) \((n\|[\w{he_chars}-]+) \$\d{1,2}\)\)', r'\1 \2', dlf)

    dlf = dlf.replace('+','')
    dlf = reformat_cop(dlf, sent)
    if dlf is None:
        return
    if 'the what' in sent:
        dlf = dlf.replace('pro:int|WHAT', '(det:art|the n|WHAT)')
    if 'a what' in sent:
        dlf = dlf.replace('pro:int|WHAT', '(det:art|a n|WHAT)')
    if 'your what' in sent:
        dlf = dlf.replace('pro:int|WHAT', '(det:poss|your n|WHAT)')
    #if sent.startswith('here \'s') or sent.startswith('here \'s') or sent.startswith('here \'s') or:
    if bool(maybe_existential_match:=re.match(r'(t?here) (is|\'s|were|are|\'re)', sent)):
        ex_np = maybe_existential_match.group(1)
        dlf = dlf.replace('cop|be-pres', 'v|equals').replace('cop|~be', 'v|equals')
        if not dlf.endswith(f' pro:exist|{ex_np}'):
            dlf = dlf + f' pro:exist|{ex_np}'
        print(dlf, sent)
    dlf = postfix_whatiss(dlf)
    if 'would' in sent:
        dlf = dlf.replace('mod|will-cond', 'mod|would')
        dlf = dlf.replace('mod|will', 'mod|would')
    dlf = dlf.replace('mod|~will', 'mod|will')
    if 'mod|~genmod' in dlf and 'would' not in sent and '\'d' not in sent:
        breakpoint()
    dlf = dlf.replace('mod|~genmod', 'mod|would')
    if dlf.startswith('hasproperty pro:dem|this'):
        breakpoint()
    return dlf

def reformat_cop(lf, sent):
    if (had_q:=lf.startswith('Q (')):
        inner = lf[3:-1]
    else:
        inner = lf
    if (had_not:=inner.startswith('not (')):
        inner = inner[5:-1]
    maybe_cop_aux = re.match(r'(aux|cop)\|(\~?)be(-past|-pres|-3s|-1s|) \(part\|', inner)
    if maybe_cop_aux is not None:
        cop_aux = maybe_cop_aux.group()
        rest = inner[len(cop_aux):]
        verb, _, rest = rest.partition('-')
        if not ( rest.startswith('presp')):
            return
        rest = rest.removeprefix('presp ')
        tense = 'past' if 'past' in cop_aux else 'pres'
        if 'I' in sent and ('\'m' in sent or 'am' in sent or 'was' in sent):
            person = '1s'
        elif 'you' in sent and ('\'re' in sent or 'are' in sent or 'were' in sent):
            person = '2s'
        else:
            person = '3s'
        if 'were' in sent or 'was' in sent:
            #if not ( tense=='past'):
                #print(f'marked cop tense wrong for {inner} {sent}')
            tense = 'past'
        elif any(w in sent for w in ['is', 'are', '\'s', '\'re', '\'m', 'am']):
            if not ( tense=='pres'):
                breakpoint()
        else:
            breakpoint()
        lf = f'cop|{tense}-{person} (v|{verb}-prog {rest}' # note rest endswith ')'
    elif bool(maybe_part_match := re.match(r'part\|([a-z]+)-presp (.*) you$', inner)):
        verb = maybe_part_match.group(1)
        obj = maybe_part_match.group(2)
        lf = f'lambda $0.v|{verb}-prog {obj} $0'
    else:
        return lf
    if had_not:
        lf = add_not(lf)
    if had_q:
        lf = add_q(lf)
    if 'aux' in lf and 'aux|have' not in lf:
        print(lf)
    if 'lambda' in all_lambda_body_splits(lf)[1]:
        breakpoint()
    if lf.count('cop')==2:
        breakpoint()
    assert iwff.is_wellformed_lf(lf)
    return lf

def sent_preproc(lf, sent):
    sent = sent.rstrip('?.! ')
    assert not sent.endswith(' ')
    if sent.startswith('because ') and 'because' not in lf: # often not in lf
        sent = sent.removeprefix('because ')
    if ARGS.tokenize_ing:
        maybe_ing_splits = [w for w in sent.split() if w.endswith('ing')]
        lf_parts = re.split(r'[,\|_\(\)\.-]', lf)
        global ings
        global falseings
        def inlf(x): return x in lf_parts and (f'part|{x}' in lf or f'v|{x}' in lf)
        for mis in maybe_ing_splits:
            stem = mis.removesuffix('ing')
            if len(stem)>1 and inlf(stem):
                #lemma = stem
                sent = sent.replace(mis, f'{stem} ing')
                ings.append(mis)
            elif len(stem)>1 and inlf(stem+'e'): # e.g. 'hiding'
                #lemma = stem+'e'
                sent = sent.replace(mis, f'{stem}e ing')
                ings.append(mis)
            elif len(stem)>1 and stem[-1]==stem[-2] and inlf(stem[:-1]):
                #lemma = stem[:-1] # e.g. 'hitting'
                sent = sent.replace(mis, f'{stem[:-1]} ing')
                ings.append(mis)
            else:
                #if mis in ings:
                    #breakpoint()
                falseings.append(mis)

                #print('\t' + sent)
    if sent in sent_fixes:
        sent = sent_fixes[sent]
    #sent = sent.replace('n \'t', ' n\'t')
    sent = sent.replace('oh','')
    sent = sent.replace('well','')
    if 'so' not in lf:
        sent = sent.replace('so','')
    sent = sent.replace(' \'t', '\'t')
    #if (had_neg:=any(w in sent for w in neg_conts)): print(sent, end=' ')
    sent = ' '.join(neg_conts.get(w,w) for w in sent.split(' '))
    #if had_neg: print('-->', sent)
    sent = sent.removeprefix('alright ')
    sent = [w for i,w in enumerate(sent.split()) if (f'co|{w}' not in lf and not(w=='we' and w not in lf)) or w=='like' and i==len(sent.split())-1] # simple hack that 'like' at end is v
    if len(sent)>0 and sent[0] == 'ʔavāl' and 'ʔavāl' not in lf:
        sent = sent[1:]
    if sent == '':
        breakpoint()
    sent = [w for w in sent if not w[0].isupper() or w.lower() in lf]
    for vocative in ('Adam', 'honey', 'dear'):
        if vocative in sent and vocative.lower() not in lf:
            sent = [w for w in sent if w!=vocative]
    if ' '.join(sent) in manual_sent_fixes:
        sent = manual_sent_fixes[' '.join(sent)].split()
    return sent

def decide_if_question(lf, sent:list, udtags:list):
    conjs = ('but', 'and', 'or')
    wh_words = ('what', 'who', 'how', 'where', 'when', 'which')
    if udtags[0] == 'CONJ' and sent[0] not in conjs:
        breakpoint()
    lf = de_q(lf)
    for conj in conjs:
        if sent[0] == conj:
            sent = sent[1:]
            if lf.startswith(conj):
                assert lf.starswith(f'{conj} (') and lf.endswith(')')
                lf = lf[len(conj)+2:-1]
    assert not bool(re.search(r'(?<!pro:per\|)you(?![a-z])', lf))
    if udtags[0] in ('AUX') or sent[0] in ('is', 'are', 'was', 'were'):
        lf = add_q(lf)
    elif sent[0] in wh_words and not (udtags[-1] in ('AUX') or sent[-1] in ('is', '\'s', 'are', '\'re', 'was', 'were')):
        lf = add_q(lf)
    return lf, sent

def apply_manual_fixes(lf, sent):
    if ' '.join(sent) in direct_take_lf_from_sents:
        print(direct_take_lf_from_sents[' '.join(sent)])
        return direct_take_lf_from_sents[' '.join(sent)]
    if lf in manual_ida_fixes.keys():
        old_lf = lf
        lf = manual_ida_fixes[lf]
        if ARGS.print_fixes:
            print(f'Fixing: {old_lf} --> {lf}')
    lf = lf.replace('cow+boy', 'cowboy')
    return lf

def postfix_whatiss(lf):
    maybe_match = re.match(r'Q \(pro:int\|WH([A-Z]+) ([a-zA-Z:\|-]+)\)', lf)
    if maybe_match is not None:
        fixed_lf = f'Q (v|equals pro:int|WH{maybe_match.group(1)} {maybe_match.group(2)})'
        #print(lf, '-->', fixed_lf)
        return fixed_lf
    maybe_match = re.match(r'Q \(pro:int\|WH([A-Z]+) (\([a-zA-Z:\|\- ]+\))\)', lf)
    if maybe_match is not None:
        fixed_lf = f'Q (v|equals pro:int|WH{maybe_match.group(1)} {maybe_match.group(2)})'
        #print(lf, '-->', fixed_lf)
        return fixed_lf
    else:
        return lf

if __name__ == '__main__':
    import argparse
    ARGS = argparse.ArgumentParser()
    ARGS.add_argument('-d', '--dset', type=str, choices=['adam', 'hagar'], default='adam')
    ARGS.add_argument('--db', type=str)
    ARGS.add_argument('--dbsent', type=str)
    ARGS.add_argument('-p', '--print-conversions', action='store_true')
    ARGS.add_argument('-f', '--print-fixes', action='store_true')
    ARGS.add_argument('--tokenize-ing', action='store_true')
    ARGS = ARGS.parse_args()

    with open(f'data/{ARGS.dset}-combined-info.json') as f:
        dset = json.load(f)

    dset_data = []
    n_excluded = 0
    with open('../CHILDES_UD2LF_2/conll/full_adam/adam.all.udv1.conllu.final') as f:
        conll = f.read().strip().split('\n\n')

    with open('../CHILDES_UD2LF_2/LF_files/full_adam/adam.all_lf.txt') as f:
        ida_lf = f.read().strip().split('\n\n')

    for dpoint in dset:
        sent, lf, udtags = dpoint['idasent'], dpoint['idalf'], dpoint['udtags']
        if 'which' in sent:
            ps, pl = manual_which_dict[sent]
            ps = ps.split()
            if pl != 'EXCL':
                dset_data.append({'lf':pl, 'words':ps})
            continue
        if 'else' in sent: # and 'else' not in lf:, always wrong in the lf
            continue
        if sent[:-2] in exclude_sents:
            n_excluded+=1
            continue
        if any(x in dpoint['idalf'] for x in exclude_lfs):
            n_excluded+=1
            continue
        if 'lambda' in all_lambda_body_splits(lf)[1]: # we're saying these are malformed
            continue
        ps = sent_preproc(lf, sent)
        if ' '.join(ps) in exclude_sents: # so exclude sents removes matches before or after preproc
            n_excluded+=1
            continue
        if len(ps) > 0 and ps[0] == 'ʔavāl':
            breakpoint()
        if ARGS.dbsent is not None and ps == ARGS.dbsent.split():
            breakpoint()
        if ps == []:
            n_excluded+=1
            continue
        #if any (w in ps for w in ['give', 'tell', 'show']):
            #breakpoint()
        pl = lf_preproc(lf, sent)
        if pl is None or pl.endswith(': ') or '_' in pl or pl=='':
            n_excluded+=1
            continue
        if not all(f'lambda {v}.' in pl for v in set(re.findall(r'\$\d{1,2}', pl))):
            continue
        assert not any(x in pl for x in exclude_lfs)
        # need to wait until after Q insertion to do this
        bare_you_reg = r'(?<!pro:per\|)you(?![a-z])'
        if re.search(bare_you_reg, pl):
            #print(pl + '-->', end=' ')
            nvn = new_var_num(pl)
            pl = f'lambda ${nvn}.' + re.sub(bare_you_reg, f'${nvn}', pl)
            pl = alpha_normalize(pl)
            #print(pl)
        pl, ps = decide_if_question(pl, ps, udtags)
        pl = apply_manual_fixes(pl, ps)
        if ARGS.print_conversions:
            print(' '.join(ps))
        if pl == 'Q ()':
            breakpoint()
        if 'lambda' in pl:
            assert pl.startswith('lambda')
        if len(pl.split())==1 and len(ps) != 1:
            print(pl, ps)
            continue
        dset_data.append({'lf':pl, 'words':ps})
    x=[' '.join(y['words']) for y in dset_data]
    y=dict(zip(*np.unique(x, return_counts=True)))
    hist_counts = sorted([(k,int(v)) for k,v in y.items()], key=lambda x:x[1], reverse=True)
    pp(hist_counts[:20])
    dset = {'data':dset_data, 'hist_counts':hist_counts}

    def printcounts(countslist):
        x,y=np.unique(countslist, return_counts=True)
        print(dict(zip(list(x), list(y))))

    printcounts(ings)
    printcounts(falseings)
    print(f'Num Excluded Points: {n_excluded}')
    print(f'Num Remaining Points: {len(dset_data)}')
    with open(f'data/{ARGS.dset}.json','w') as f:
        json.dump(dset,f)
    with open(f'data/{ARGS.dset}_no_commas.txt','w') as f:
        for dpoint in dset_data:
            sent, lf = ' '.join(dpoint['words']), dpoint['lf']
            f.write(f'Sent: {sent}\n')
            f.write(f'Sem: {lf}\n\n')
