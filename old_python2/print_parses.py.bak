# This contains a function that takes the parse chart and prints 
# every possible parse in it.
# May extend this to allow us to search for a specific parse although
# this might be pointless as we can just search in the output file.
import copy

def parse_cross_prod(left_parses,right_parses):
	parses = []
	for lp in left_parses:
		for rp in right_parses:
			p = copy.copy(lp)
			p.extend(rp)
			parses.append(p)
	return parses
	
def return_parses(sentence_chart,entry):
	parses = []
	parses.append([entry.cat.key])
	for pair in entry.children:
		left_syn = pair[0][0]
		right_syn = pair[1][0]
		left_parses = return_parses(sentence_chart,sentence_chart[pair[0][3]-pair[0][2]][pair[0]])
		right_parses = return_parses(sentence_chart,sentence_chart[pair[1][3]-pair[1][2]][pair[1]])
		child_parses = parse_cross_prod(left_parses,right_parses)
		parses.extend(child_parses)
	return parses	
		#parse = return_parses(sentence_chart[pair[0][3]-pair[0][2]][pair[0]]).extend(return_parses(sentence_chart[pair[1][3]-pair[1][2]][pair[1]]))
		#parses.append(parse)	
def print_parses(sentence_chart,output):
	top_key = sentence_chart[len(sentence_chart)].keys()[0]
	top = sentence_chart[len(sentence_chart)][top_key]
	parses = return_parses(sentence_chart,top)
	for p in parses:
		print >> output,p
		slashes = 0
		for bit in p:
			slashes += bit.count('/')
			slashes += bit.count('\\')
		if not slashes == len(p) -1:
			print >> output,'not right no of slashes'

