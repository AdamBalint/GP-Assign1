import operator
import math
import random
from random import shuffle
import GP_Graph as gpg
import scipy as sp

import KFold

import numpy as np


from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

fold_k = 10
current_k = 0
folds = []

# Opens and reads in the data
all_data = []
f = open('Dataset/pima-indians-diabetes/pima-indians-diabetes.data', 'r')
for line in f:
    tmp = line.strip('\n').split(',')
    tmp = [float(x) for x in tmp]
    tmp2 = (tmp)
    all_data.append(tmp)
    print(tmp2)

f.close()

# defines how to split the data into testing and training set
def splitData(all_data, fold_k):
    train_percent = 0.4#0.5 # 0.4-0.5
    test_percent = 0.6 #0.6-0.5


    shuffle(all_data)

    train_data = all_data[:int(len(all_data)*train_percent)]
    test_data = all_data[int(len(all_data)*train_percent):]

    folds = []

    # splits the data into k folds as well
    num_per_fold = int(len(train_data)/fold_k)
    for i in range(fold_k-1):
        folds.append(train_data[i*num_per_fold:(i+1)*num_per_fold])

    folds.append(train_data[(fold_k-1)*num_per_fold:])

    num_in_test = sum(len(fold) for fold in folds)
    print ("Num Folds: " + str(fold_k) + "\t Num per fold: " + str(num_per_fold))
    print ("Length of Original: " + str(len(train_data)) + "\t Length of Folds: " + str(num_in_test))

    return folds, test_data






# Definition of the protected div
def protectedDiv(left, right):
    # if the number is close to 0, treat is as 0 to prevent infinity
    try:
        if (right < 0.000000000001):
            return 1
        return left / right
    except ZeroDivisionError:
        return 1
    except RuntimeWarning:
        print("Left: " + str(left) + "\tRight: " + str(right))
        return 1

# defines the absolute quare root
def abs_sqrt(num):
    return math.sqrt(abs(num))

def cbrt(num):
    return sp.special.cbrt(num)

# defines sin using radians
def sin(num):
    try:
        n2 = math.radians(num)
        return math.sin(n2)
    except ValueError:
        print("Infinity Warning")
        print("Num: " + str(num))

# defines cosine using radians
def cos(num):
    try:
        n2 = math.radians(num)
        return math.cos(n2)
    except ValueError:
        print("Infinity Warning")
        print("Num: " + str(num))

# Defines the modulo operator
def modulo(n1, n2):
    if (n2 == 0):
        return n1
    return n1%n2

# Defines distance calculation using 4 inputs
def dist(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

# Defines an if statement comparing to 0
def ifStatement(n1, n2, n3):
    if (n1 < 0):
        return n2
    else:
        return n3

# defines an if statement comparing 2 numbers
def compStatement(n1, n2, n3, n4):
    if (n1 < n2):
        return n3
    else:
        return n4

# calculates the determinant of a 2x2 matrix made from 4 inputs
def det2x2(n11, n12, n21, n22):
    return (n11*n22-n12*n21)


# allows to set the test number to use, and the base output name
test_num = 1;
name = "Experiment-" + str(test_num)




# set up GP parameters
# 8 input variables
pset = gp.PrimitiveSet("MAIN", 8)
# add the addition operator. Has 2 inputs
# sets up the rest of the operators to use
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)

# include other operators based on test number
if (test_num > 1):
    pset.addPrimitive(sin,1)
if (test_num > 3):
    pset.addPrimitive(cbrt, 1)
if (test_num > 4):
    pset.addPrimitive(round, 1)
if (test_num > 5):
    pset.addPrimitive(ifStatement,3)
if (test_num > 2):
    pset.addPrimitive(math.tanh,1)

# unused operators
#pset.addPrimitive(cos,1)
#pset.addPrimitive(abs_sqrt, 1)
#pset.addPrimitive(max, 2)
#pset.addPrimitive(min, 2)
#pset.addPrimitive(math.floor, 1)
#pset.addPrimitive(math.ceil, 1)
#pset.addPrimitive(modulo, 2)
#pset.addPrimitive(dist,4)
#pset.addPrimitive(compStatement,4)
#pset.addPrimitive(det2x2,4)

# add the posibility of a constant from -1 to 1
pset.addEphemeralConstant("const", lambda: random.uniform(-1, 1))

# rename inputs to more readable names
pset.renameArguments(ARG0="pregnancies")
pset.renameArguments(ARG1="glucose")
pset.renameArguments(ARG2="bPressure")
pset.renameArguments(ARG3="skinThickness")
pset.renameArguments(ARG4="insulin")
pset.renameArguments(ARG5="bmi")
pset.renameArguments(ARG6="pedigreeFunc")
pset.renameArguments(ARG7="age")

# state that it is a maximization problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# set the individuals language and fitness type
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
# set tree generation parameters
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
# Set up individuals and populations and set up the compilation process
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# define the evaluation function
# counts the number of correct classifications
# Method must return an array
def evalFunc(individual, current_k, fold_k, folds):
    func = toolbox.compile(expr=individual)
    total = 1.0

    for i in range(fold_k):
        if (i == current_k):
            continue

        for test in folds[i]:
            total += int((0 if (func(*test[:8])) < 0 else 1) == test[8])

        current_k = (current_k+1)%fold_k
    return total,

# define the evaluation for the testing
# Identical to above, but does more logging
def testEval(individual, test_data):
    tp, tn, fp, fn = 0,0,0,0
    func = toolbox.compile(expr=individual)
    total = 0
    for test in test_data:
        res = int((0 if (func(*test[:8])) < 0 else 1) == test[8])
        if (res == 1):
            if (test[8] == 1):
                tp += 1
            else:
                tn += 1
        else:
            if (test[8] == 1):
                fn += 1
            else:
                fp += 1

        total += res
    return [total/len(test_data), tp, tn, fp, fn]

# register the evaluation function
toolbox.register("evaluate", evalFunc, current_k=current_k, fold_k=fold_k, folds=folds)

# set up GP parameters
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# Set up logging
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
# register methods for calculating various statistics
mstats.register("mean", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)

# open file summary
f_avg = open('Logs/'+name+'-avg.txt', 'w')
avg_vals = []

# run the test 20 times
for i in range(1):
    # split the data each time and register the evaluation function using the correct data
    folds, test_data = splitData(all_data, fold_k)
    toolbox.register("evaluate", evalFunc, current_k=current_k, fold_k=fold_k, folds=folds)

    # generate the populations
    pop = toolbox.population(n=1000)
    # holds the n best individuals
    hof = tools.HallOfFame(1)
    print ("Run: " + str(i))
    pop, logs = algorithms.eaSimple(pop, toolbox, 0.9, 0.1, 60, stats=mstats, halloffame=hof, verbose=True)
    # Open file to log the results
    f = open('Logs/'+name+'-logs-' + str(i) +'.txt', 'w')
    f.write(str(logs))
    f.close()
    expr = hof[0]
    # Print and store the testing results for the best solution
    print("fitness: " + str(testEval(expr, test_data)))
    avg_vals.append("\t".join(str(s) for s in testEval(expr, test_data)))

# write results to the file
f_avg.write(str("\n".join(str(s) for s in avg_vals)))
f_avg.close()
