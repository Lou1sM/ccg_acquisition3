
def assemble_nodes_to_tree(nodes):
    for k,v in nodes.items():
        if k=='1':
            continue
        parent_idx = v['idx'][:-2]+'1'
        if v['idx'][-2] == '0':
            nodes[parent_idx]['left_child'] = v
        else:
            assert v['idx'][-2] == '2'
            nodes[parent_idx]['right_child'] = v
    lengths = set([len(k) for k in nodes.keys()]) # n_levels in the tree
    tree = [[v for k,v in nodes.items() if len(k) == l] for l in lengths]
    return tree

gts = {}

nodes = {}
nodes['1'] = {"idx": "1", "sem_cat": "Sq", "syn_cat": "Sq", "rule": "fwd_app", "shell_lf": "Q (const (quant noun))", "lf": "Q (talk (a (lake)))", "words": "does a lake talk"}
nodes['01'] = {"idx": "01", "sem_cat": "Sq|S", "syn_cat": "Sq/S", "rule": "leaf", "shell_lf": "lambda $0.Q ($0)", "lf": "lambda $0.Q ($0)", "words": "does"}
nodes['21'] = {"idx": "21", "sem_cat": "S", "syn_cat": "S", "rule": "bck_app", "shell_lf": "const (quant noun)","lf": "talk (a (lake))", "words": "a lake talk"}
nodes['201'] = {"idx": "201", "sem_cat": "NP", "syn_cat": "NP", "rule": "fwp_app", "shell_lf": "quant noun", "lf": "a (lake)", "words": "a lake"}
nodes['221'] = {"idx": "221", "sem_cat": "S|NP", "syn_cat": "S\\NP", "rule": "leaf", "shell_lf": "lambda $0.const $0", "lf": "lambda $0.talk $0", "words": "talk"}
nodes['2001'] = {"idx": "2001", "sem_cat": "NP|N", "syn_cat": "NP/N", "rule": "leaf", "shell_lf": "lambda $0.quant $0", "lf": "lambda $0.a $0", "words": "a"}
nodes['2021'] = {"idx": "2021", "sem_cat": "N", "syn_cat": "N", "rule": "leaf", "shell_lf": "noun", "lf": "lake", "words": "lake"}
gts['does_a_lake_talk'] = assemble_nodes_to_tree(nodes)

nodes = {}
nodes['1'] = {"idx": "1", "sem_cat": "S", "syn_cat": "S", "rule": "bck_app", "shell_lf": "const const const", "lf": "boarder virginia texas", "words": "virginia boarders texas"}
nodes['01'] = {"idx": "01", "sem_cat": "NP", "syn_cat": "NP", "rule": "leaf", "shell_lf": "const", "lf": "const", "words": "virginia"}
nodes['21'] = {"idx": "21", "sem_cat": "S\\NP", "syn_cat": "S\\NP", "rule": "fwd_app", "shell_lf": "lambda $0.const $0 const", "lf": "lambda $0.boarder $0 texas", "words": "boarders texas"}
nodes['201'] = {"idx": "201", "sem_cat": "S|NP|NP", "syn_cat": "S\\NP/NP", "rule": "leaf", "shell_lf": "lambda $0.lambda $1.const $1 $0", "lf": "lambda $0.lambda $1.boarder $1 $0", "words": "boarders"}
nodes['221'] = {"idx": "221", "sem_cat": "NP", "syn_cat": "NP", "rule": "leaf", "shell_lf": "const", "lf": "texas", "words": "texas"}
gts['virginia_boarders_texas'] = assemble_nodes_to_tree(nodes)
