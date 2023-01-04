import json
import numpy as np

with open('data/preprocessed_geoqueries.json') as f:
    d = json.load(f)

nps = d['np_list']

transitives = {'longer': ('is_longer_than', False), 'lower': ('is_lower_than', False), 'low_point': ('is_a_low_point_of', False), 'area': ('has_area', True), 'capital': ('is_the_captial_of', False), 'next_to': ('is_next_to', False), 'len': ('has_length', True), 'population': ('has_population', True), 'size': ('has_size', True), 'traverse': ('traverses', False), 'higher': ('is_higher_than', False), 'high_point': ('is_a_high_point_of', False), 'density': ('has_density', True), 'elevation': ('has_elevation', True), 'loc': ('is_in', False)}

intransitives = {'state': 'is_a_state', 'capital': 'is_a_capital', 'place': 'is_a_place', 'lake': 'is_a_lake', 'mountain': 'is_a_mountain', 'city': 'is_a_city', 'river': 'is_a_river'}

nums = ['one','two','three','four','five','six','seven','eight','nine','ten']

def make_random_dpoint():
    subj = np.random.choice(nps)
    verb = np.random.choice(list(transitives) + list(intransitives))
    if verb in intransitives:
        return {'words': ' '.join([subj, intransitives[verb]]).split(), 'parse':f'{verb} {subj}'}
    verb_str,requires_num = transitives[verb]
    if requires_num:
        obj_str = np.random.choice(nums)
        obj = nums.index(obj_str)
    else:
        obj = obj_str = np.random.choice(nps)
    return {'words': ' '.join([subj ,verb_str ,obj_str]).split(), 'parse':f'{verb} {subj} {obj}'}

dpoints = [make_random_dpoint() for _ in range(1000)]
processed_dset = {'np_list':nps, 'intransitive_verbs':list(intransitives),
                    'transitive_verbs': list(transitives), 'data':dpoints}
with open('data/simple_dset.json','w') as f:
    json.dump(processed_dset,f)
