from get_hagar_ud_tags import parse_conll
import json



for dset_name in ('adam', 'hagar'):
    print(dset_name)
    if dset_name=='adam':
        conll_fpath = '../CHILDES_UD2LF_2/conll/full_adam/adam.all.udv1.conllu.final'
    else:
        conll_fpath = '../CHILDES_UD2LF_2/conll/full_hagar/hagar.all.udv1.conllu.current'
    with open(conll_fpath) as f:
        conll = f.read().strip().split('\n\n')

    with open(f'../CHILDES_UD2LF_2/LF_files/full_{dset_name}/{dset_name}.all_lf.txt') as f:
        ida_lf = f.read().strip().split('\n\n')

    print(len(conll), len(ida_lf))

    if dset_name=='adam':
        conll_sents = [' '.join(parse_conll(c)[0]) for c in conll] # take lemmas bc they're what's in sent
    else:# take lemmas bc they're mixed up with word forms for Hagar
        conll_sents = [' '.join(parse_conll(c)[1]) for c in conll]
    conll_sents = [x.replace(' -s','').replace('-', ' ').replace('_', ' ') for x in conll_sents]

    ida_sents = [x.split('\n')[0][6:].strip().replace('  ',' ') for x in ida_lf]
    ida_sents_ = [x.rstrip(' .?!') for x in ida_sents]
    ida_lfs = [x.split('\n')[1][5:].strip() for x in ida_lf]
    combined_info = []
    for i, (sent,ilf) in enumerate(zip(ida_sents, ida_lfs)):
        print(i)
        try:
            corr_conll = conll[conll_sents.index(sent.replace('_',' '))]
        except ValueError:
            print([x for x in conll_sents if x.startswith(sent[:10])])
        #assert len(sent.split())== len(corr_conll.split('\n'))
        word_forms, lemmas, udtags, chiltags = parse_conll(corr_conll)
        combined_info.append({'lemmas': lemmas,
                              'chiltags': chiltags,
                              'udtags': udtags,
                              'idasent':sent,
                              'idalf':ilf})
    with open(f'data/{dset_name}-combined-info.json','w') as f:
        json.dump(combined_info, f)
