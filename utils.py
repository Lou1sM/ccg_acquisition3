import re


def is_balanced_nums_brackets(s):
    return len(re.findall(r'\(',s)) == len(re.findall(r'\)',s))

def outermost_first_bracketted_chunk(s):
    num_open_brackets = 0
    has_been_bracketed = False
    for i,c in enumerate(s):
        if c == '(':
            num_open_brackets += 1
            has_been_bracketed = True
        elif c == ')':
            num_open_brackets -= 1
        if num_open_brackets == 0 and has_been_bracketed:
            assert is_balanced_nums_brackets(s[:i+1])
            assert is_balanced_nums_brackets(s[i+1:])
            return s[:i+1], s[i+1:]
    breakpoint()


def split_respecting_parentheses(s,sep=' '):
    """Only make a split when there are no open parentheses. Second return is
    a bool indicating if there are superfluous surrounding parentheses."""
    #while s.startswith('(') and s.endswith(')'):
        #s = s[1:-1]
    num_open_brackets = 0
    split_points = [-1]
    for i,c in enumerate(s):
        if c == sep and num_open_brackets == 0:
            split_points.append(i)
        elif c == '(':
            num_open_brackets += 1
        elif c == ')':
            num_open_brackets -= 1
    #if split_points == [-1] and ',' in s and s.startswith('(') and s.endswith(')'):
        #splits, _ =  split_respecting_parentheses(s[1:-1],sep=sep)
        #return splits, True
    split_points.append(len(s))
    splits = [s[split_points[i]+1:split_points[i+1]] for i in range(len(split_points)-1)]
    return splits

def is_bracketed(s):
    return s.startswith('(') and s.endswith(')')
