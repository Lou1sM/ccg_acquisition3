import re

adjFile = open('./logical_forms/2Adam_adj_lfs', "w")
detFile = open('./logical_forms/2Adam_det_lfs', "w")
nounFile = open('./logical_forms/noun_lfs', "w")
prepFile = open('./logical_forms/2Adam_prep_lfs', "w")
verbFile = open('./logical_forms/2Adam_verb_logical_forms', "w")

current_sent = ''
adjDict = {}
detDict = {}
nounDict = {}
prepDict = {}
verbDict = {}

def levenshtein_distance(word1, word2):
    """
    Calculates the edit distance between word1 (longer) and word2 (shorter)
    :param word1
    :param word2
    :return edit distance
    """
    if len(word1) < len(word2):
        return levenshtein_distance(word2, word1)
    if len(word2) == 0:
        return len(word1)

    previous_row = range(len(word2) + 1)
    for i, c1 in enumerate(word1):
        current_row = [i + 1]
        for j, c2 in enumerate(word2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def putInDict(d, items, sentence):
    for x in items:
        name  = index_regex.split(x)[0]
        nameNoPOSNoMorph = name.split('|')[1].split('-')[0]
        distances = [levenshtein_distance(nameNoPOSNoMorph, w) for w in sentence]
        word = sentence[distances.index(min(distances))]
        if name not in d:
            d[word] = name

index_regex = re.compile("_\d+")
for i in range(1, 56):
    for line in open('./trainFiles/Adam{}_lf.txt'.format(str(i)), "r"):
        if line.startswith("Sent: "):
            current_sent = line.rstrip().split("Sent: ")[1].split()
        if line.startswith("Sem: "):
            l = line.rstrip().split("Sem: ")[1]
            adjectives = re.compile("(?<=\W)adj\|[\w-]+").findall(l)
            possesives = re.compile("(?<=\W)det:poss\|[\w-]+").findall(l)
            other_possesives = re.compile("(?<=\W)pro:poss\|[\w-]+").findall(l)
            determiners = re.compile("(?<=\W)det:art\|[\w-]+").findall(l)
            quantifiers = re.compile("(?<=\W)qn\|[\w-]+").findall(l)
            nouns = re.compile("(?<=\W)n\|[\+\w-]+").findall(l)
            prepositions = re.compile("(?<=\W)prep\|[\w-]+").findall(l)
            verbs = re.compile("(?<=\W)v\|[\w-]+").findall(l)
            participles = re.compile("(?<=\W)part\|[\w-]+").findall(l)
            putInDict(adjDict, adjectives, current_sent)
            putInDict(detDict, possesives+other_possesives+determiners, current_sent)
            putInDict(nounDict, nouns, current_sent)
            putInDict(prepDict, prepositions, current_sent)
            putInDict(verbDict, verbs+participles, current_sent)


for d, f in  zip([adjDict, detDict, nounDict, prepDict, verbDict], [adjFile, detFile, nounFile, prepFile, verbFile]):
# for d, f in [(nounDict, nounFile)]:
    for w, lf in d.items():
        f.write(w + " " + lf + "\n")


