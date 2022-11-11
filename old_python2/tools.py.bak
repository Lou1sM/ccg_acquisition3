###########################################
# SMALL SCRIPTS USED MULTIPLE TIMES.      #
###########################################
from math import log
from math import exp
# there are other scripts in old versions # 
# of the code that might be worth looking # 
# at.


###########################################
# This function gives all possible ordered#
# sets of length n of members of the list #
# items.                                  #
###########################################
def xcombinations(items, n):
	if n==0: yield []
	else:
		for i in xrange(len(items)):
			for cc in xcombinations(items[:i]+items[i+1:],n-1):
				yield [items[i]]+cc

				
###########################################



###########################################
# This function is used to generate the   #
# set of bracketings of the sentences     #
# needed to support the semantics.        #
###########################################
def give_comb_numbs(len_sen, len_sem):
	comb_numbs = []
	i = len_sen - len_sem +1
	if len_sem == 1:
		comb_numbs.append([i])
	else:
		while i >= 1:
			comb_numbs_temp = []
			comb_numbs_temp = give_comb_numbs(len_sen-i, len_sem-1)
			if len(comb_numbs_temp) == 0:
				print 'ERROR - give_comb_numbs'
			else:
				for item in comb_numbs_temp:
					item.append(i)
				comb_numbs.extend(comb_numbs_temp)
			i = i-1
	return comb_numbs

###########################################



###########################################
# This function generates the possible    #
# wordsets for a sentence with number of  #
# atomic semantic components given by     #
# max_sem.						          #
###########################################
def generate_wordset(sentence,max_sem):
	wordset = {}
	for i in range(1, max_sem+1):
		wordset[i] = []
		comb_numbs = give_comb_numbs(len(sentence), i)
		for comb in comb_numbs:
			words = []
			j = 0
			for item in comb:
				word = ''
				for k in range(item):
					if k==0: word = sentence[j]
					else: word = word+' '+sentence[j]
					j=j+1
				words.append(word)
			wordset[i].append(words)
	return wordset
	
###########################################



###########################################
# Check for loops in the semantics.       #
###########################################
def check_for_self_loop(semantic_components,head,seen):
	seen.append(head)
	for dep in semantic_components[head].Dependents:
		if dep in seen:
			return True
		elif check_for_self_loop(semantic_components,dep,seen):
			return True
	return False
###########################################


###########################################
# Log Sum (log(x),log(y)) = 			  #
###########################################
def log_sum(a,b):	
	if a==-inf:	return b;
	elif b==-inf: return a 
	else:
		if(a>b): return a + log(1+exp(b-a))
		else: return b + log(1+exp(a-b));

###########################################


###########################################
# Log Diff 	log(a-b)			          #
###########################################
def log_diff(a,b):	
	if b==-inf:	return a;
	elif a<b:
		print "a = ",a
		print "b = ",b
		return float("NaN")
	elif a==b:
		return -inf
	else:
		return a + log(1-exp(b-a))

inf = float("inf")



###########################################                                                                                                         # permutations                    #                                                                                                                 ###########################################                                                                                                          
def permutations(lset):
	perms = []
	if len(lset)==1: return [lset]
	i = 0
	for l in lset:
		pset = permutations(lset[:i]+lset[i+1:])
		for p in pset:
			pl = [l]
			pl.extend(p)
			perms.append(pl)
		i+=1
	return perms
