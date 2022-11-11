from exp import *
from predicate import *
from constant import *
from conjunction import *
from neg import *
from qMarker import *
import re



def makeExp(predString, expString, expDict):
    if predString in expDict:
        e, coveredString = expDict[predString]
        expStringRemaining = expString if not coveredString else expString.split(coveredString)[1]
        return e, expStringRemaining

    name = predString.strip().rstrip()
    nameNoIndex = re.compile("_\d+").split(name)[0]
    pos = name.split("|")[0]
    args, expStringRemaining = extractArguments(expString, expDict)
    argTypes = [x.type() for x in args]
    numArgs = len(args)

    if numArgs == 0:
        e = constant(nameNoIndex,numArgs,argTypes,pos)
        e.makeCompNameSet()
    elif numArgs in [2,3]:
        is_quant = isQuant(args)
        if pos in ['conj', 'coord']:
            e = conjunction()
            e.setType(name)
        elif is_quant:
            e = predicate(nameNoIndex,numArgs,argTypes,pos,bindVar=True,args=args)
        else:
            e = predicate(nameNoIndex,numArgs,argTypes,pos,args=args)
    else:
        e = predicate(nameNoIndex,numArgs,argTypes,pos,args=args)

    for i, arg in enumerate(args):
        e.setArg(i,arg)
    e.setString()

    expDict[predString] = [e, getCoveredString(expString, expStringRemaining)]
    return e, expStringRemaining

def getCoveredString(expString, expStringRemaining):
    if expString and not expStringRemaining:
        coveredString = expString
    elif expString:
        coveredString = expString.split(expStringRemaining)[0]
    else:
        coveredString = ""
    return coveredString

def isQuant(args):
    quant_with_var = args[0].__class__ == variable and args[0] in args[1].allSubExps()
    quant_with_const = args[0].__class__ == constant and any([args[0].equals(x) for x in args[1].allSubExps()])
    if len(args) == 2:
        return quant_with_var or quant_with_const
#    if len(args) == 3:
#        if args[2].__class__ not in [variable, eventMarker]:
#            return False
#        else:
#            return quant_with_var or quant_with_const
    else:
        return False

def makeExpWithArgs(expString, expDict):
    print "making ",expString
    is_lambda = expString[:6]=="lambda"
    arguments_present = -1<expString.find("(")<expString.find(")")
    no_commas = expString.find(",")==-1
    commas_inside_parens = -1<expString.find("(")<expString.find(",")

    if is_lambda:
        e, expStringRemaining = makeLambda(expString, expDict)
    elif arguments_present and (commas_inside_parens or no_commas):
        e, expStringRemaining = makeComplexExpression(expString, expDict)
    else:
        e, expStringRemaining = makeVarOrConst(expString, expDict)

    return e,expStringRemaining

def makeLambda(expString, expDict):
    vname = expString[7:expString.find("_{")]
    tstring = expString[expString.find("_{")+2:expString.find("}")]
    v = variable(None)
    t = semType.makeType(tstring)
    v.t = t
    if tstring == "r":
        v.isEvent = True
    expDict[vname] = v
    v.name = vname
    expString = expString[expString.find("}.")+2:]
    (f,expStringRemaining) = makeExpWithArgs(expString, expDict)
    e = lambdaExp()
    e.setFunct(f)
    e.setVar(v)
    e.setString()
    return e,expStringRemaining

def makeComplexExpression(expString, expDict):
    predstring, expString = expString.split("(",1)
    if predstring in ["and", "and_comp", "not", "Q"]:
        e, expStringRemaining = makeLogExp(predstring, expString, expDict)
    elif predstring[0]=="$":
        e, expStringRemaining = makeVars(predstring, expString, expDict)
    else:
        e, expStringRemaining = makeExp(predstring, expString, expDict)
    if e is None:
        print "none e for |" + predstring + "|"
    return e,expStringRemaining

def makeVarOrConst(expString, expDict):
    if expString.__contains__(",") and expString.__contains__(")"):
        constend = min(expString.find(","),expString.find(")"))
    else:
        constend = max(expString.find(","),expString.find(")"))
    if constend == -1:
        constend = len(expString)
    conststring = expString[:constend]
    if conststring[0]=="$":
        e, expStringRemaining = makeVars(conststring, expString[constend:], expDict, parse_args=False)
    else:
        e, expStringRemaining = makeExp(conststring, "", expDict)
    return e,expStringRemaining

def extractArguments(expString, expDict):
    finished = False if expString else True
    numBrack = 1
    i = 0
    j = 0
    arglist = []
    while not finished:
        if numBrack==0:
            finished = True
        elif expString[i] in [",",")"] and numBrack==1:
            if i>j:
                a, _ = makeExpWithArgs(expString[j:i], expDict)
                if not a:
                    error("cannot make exp for "+expString[j:i])
                arglist.append(a)
            j = i+1
            if expString[i]==")": finished = True

        elif expString[i]=="(": numBrack+=1
        elif expString[i]==")": numBrack-=1
        i += 1
    return arglist, expString[i:]


def makeVars(predstring,expString,vardict,parse_args=True):
    if not vardict.has_key(predstring):
        if "_{" in predstring:
            vname = predstring[:predstring.find("_{")]
            tstring = predstring[predstring.find("_{")+2:predstring.find("}")]
        else:
            # variable bound by a quantifier
            vname = predstring
            tstring = 'e'
        t = semType.makeType(tstring)
        e = variable(None)
        e.t = t
        e.name = vname
        vardict[vname] = e
    else:
        e = vardict[predstring]

    if e.numArgs == 0 and parse_args:
        args, expString = extractArguments(expString, vardict)
        for arg in args:
            e.addArg(arg)
    return e, expString


def makeLogExp(predstring,expString,vardict):
    e = None
    if predstring=="and" or predstring=="and_comp":
        e = conjunction()
        args, expString = extractArguments(expString, vardict)
        for i, arg in enumerate(args):
            e.setArg(i,arg)
        e.setString()

    elif predstring=="not":
        negargs = []
        while expString[0]!=")":
            if expString[0]==",":
                expString = expString[1:]
            a, expString = makeExpWithArgs(expString,vardict)
            negargs.append(a)
        else:
            e = neg(negargs[0], len(negargs))
            if len(negargs) > 1:
                e.setEvent(negargs[1])
        expString = expString[1:]
        e.setString()

    elif predstring == "Q":
        qargs = []
        while expString[0]!=")":
            if expString[0]==",":
                expString = expString[1:]
            a, expString = makeExpWithArgs(expString,vardict)
            qargs.append(a)
        if len(qargs)!=1:
            error(str(len(qargs))+"args for Q")
        else:
            e = qMarker(qargs[0])
        expString = expString[1:]

    return e,expString


#IDA: generic expression parsing code; use if simplification of the old code doesn't work out
# def parseExp(expString):
#     tokens = separate_parens(expString).split()
#     expList, _ = parse(tokens, 0)
#     return expList
#
#
# def parse(expression, index):
#     # expression can be a lambda expression or a function application
#     # returns an expression an the index of the first element after the closing ) of the expression
#     if expression[index] == "lambda":
#         typed_variable = expression[index+1]
#         body, next_index = parse(expression, index+3)
#         lam_args = [typed_variable, body]
#         return ["lambda", lam_args], next_index+1
#     else:
#         pred = expression[index]
#         args, next_index = parse_arguments(expression, [], index+2)
#         return [pred, args], next_index
#
#
# def parse_arguments(arg_string, args, next_index):
#     next_piece = arg_string[next_index]
#     if next_piece == ")":
#         return args, next_index+1
#     elif next_piece != "(":
#         if arg_string[next_index+1] != "(":
#             args.append(next_piece)
#             return parse_arguments(arg_string, args, next_index+1)
#         else:
#             children_args, new_next_index = parse_arguments(arg_string, [], next_index+2)
#             arg = [next_piece, children_args]
#             args.append(arg)
#             return parse_arguments(arg_string, args, new_next_index)
#     else:
#         arg, new_next_index = parse(arg_string, next_index)
#         args.append(arg)
#         return parse_arguments(arg_string, args, new_next_index)
#
#
# def separate_parens(expression):
#     new_expression = []
#     for char in expression:
#         if char == "(":
#             new_expression.append(" ( ")
#         elif char == ")":
#             new_expression.append(" ) ")
#         elif char == ".":
#             new_expression.append(" . ")
#         elif char == ",":
#             new_expression.append(" ")
#         else:
#             new_expression.append(char)
#     return ''.join(new_expression)
