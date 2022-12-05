import re


def is_balanced_nums_brackets(s):
    return len(re.findall(r'\(',s)) == len(re.findall(r'\)',s))

def outermost_first_bracketted_chunk(s):
    """Similar to the first argument returned by split_respecting_brackets,
    but here we don't need a separator, just split as soon as brackest are
    closed, e.g. f(x,y)g(z) --> f(x,y)
    """

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


def split_respecting_brackets(s,sep=' '):
    """Only make a split when there are no open brackets."""
    num_open_brackets = 0
    split_points = [-1]
    for i,c in enumerate(s):
        if c == sep and num_open_brackets == 0:
            split_points.append(i)
        elif c == '(':
            num_open_brackets += 1
        elif c == ')':
            num_open_brackets -= 1
    split_points.append(len(s))
    splits = [s[split_points[i]+1:split_points[i+1]] for i in range(len(split_points)-1)]
    return splits

def is_bracketed(s):
    return s.startswith('(') and s.endswith(')')

def all_sublists(x):
    if len(x) == 0:
        return [[]]

    recursed = all_sublists(x[1:])
    return recursed + [[x[0]]+item for item in recursed]
