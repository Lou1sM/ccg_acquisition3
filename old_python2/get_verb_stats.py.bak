import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18})

import verb_repo, sys, collections
import cPickle as pickle
import pdb
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load_repo(fn):
    f = open(fn)
    repo = pickle.load(f)
    f.close()
    return repo

def thresh_index(val,thresholds):
    for ind,thresh in enumerate(thresholds):
        if val < thresh:
            return ind
    return len(thresholds)

def thresh_index(val,thresholds):
    for ind,thresh in enumerate(thresholds):
        if val < thresh:
            return ind
    return len(thresholds)

def bin_by_freq(repo,thresholds1,thresholds2):
    instance_list = repo.verb_prob_instances()  # (verb,#occurrance,first occurance,total number of occurrances,Pr(syn,w|LF))
    print(len(instance_list))
    bins = collections.defaultdict(list) # map (bin index,#occurrance) --> mean prob
    for v in instance_list:
        bin_ind = (thresh_index(v[2],thresholds1),thresh_index(v[3],thresholds2))
        bins[(bin_ind,v[1])] = bins[(bin_ind,v[1])] + [v]
    binned_probs = collections.defaultdict(list)
    binned_verbs = collections.defaultdict(list)
    for bin_ind,verb_instances in bins.items():
        binned_probs[bin_ind[0]] = binned_probs[bin_ind[0]] + \
            [(bin_ind[1],1.0 * sum([v[4] for v in verb_instances]) / len(verb_instances))]
        binned_verbs[bin_ind[0]] = binned_verbs[bin_ind[0]] + [(v[0],v[1],v[3]) for v in verb_instances]
    return binned_probs, binned_verbs
        
def draw_graph(results,output_file):
    """
    Receives a list of lists of pairs of (x value, y value).
    Plots the graph
    """
    legend_titles = []
    index = 0
    for n,res in results.items():
        if res == []:
            continue
        #legend_titles.append('bins_'+str(n))
        res.append((0,0))
        arr = np.array(res)
        arr = arr[arr[:,0].argsort(),]
        if index == 0:
            plt.plot(arr[:,0],arr[:,1],'--',linewidth=3.0,alpha=0.8)
        else:
            plt.plot(arr[:,0],arr[:,1],'-',linewidth=3.0)            
        index += 1
    num_bins = len(results)
    plt.legend(['frequent','infrequent'],'best',prop={'size':12})
    plt.xlabel('#Seen Examples of Verb')
    plt.ylabel('Avg. CPP')
    #plt.suptitle('Learning of Frequent vs. Infrequent Transitives')
    
    plt.xlim(0,6)
    plt.ylim([0.0,1.0])
    #plt.show()
    plt.savefig(output_file)
    #plt.ylim(-0.1,1.1)



def main(fns,output_file):
    repo = verb_repo.VerbRepository()
    for fn in fns:
        new_repo = load_repo(fn)
        print(len(repo.verb_prob_instances()))
        repo.update(new_repo)
    bins, binned_verbs = bin_by_freq(repo,[],[10])
    for k,v in sorted(binned_verbs.items()):
        print(str(k)+'\t'+' '.join(set([str((x[0],x[2])) for x in v])))
    draw_graph(bins,output_file)

if __name__ == '__main__':
    main([x for x in sys.argv[1].split(':') if x != ''],sys.argv[2])
