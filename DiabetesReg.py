import operator
import math
import random
from random import shuffle
import GP_Graph as gpg
import scipy as sp

import KFold

import numpy as np

# increase population size
# plot graph size
# try if statements
# print logs to file
# Due Oct. 17th
# 70+ good
# plot fitnesses
# plot fitness vs tree size


from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

fold_k = 10
current_k = 0
folds = []

all_data = []
f = open('Dataset/pima-indians-diabetes/pima-indians-diabetes.data', 'r')
for line in f:
    tmp = line.strip('\n').split(',')
    tmp = [float(x) for x in tmp]
    tmp2 = (tmp)
    all_data.append(tmp)
    print(tmp2)

f.close()


def splitData(all_data, fold_k):
    train_percent = 0.4#0.5 # 0.4-0.5
    test_percent = 0.6 #0.6-0.5


    shuffle(all_data)

    train_data = all_data[:int(len(all_data)*train_percent)]
    test_data = all_data[int(len(all_data)*train_percent):]

    folds = []

    num_per_fold = int(len(train_data)/fold_k)
    for i in range(fold_k-1):
        folds.append(train_data[i*num_per_fold:(i+1)*num_per_fold])

    folds.append(train_data[(fold_k-1)*num_per_fold:])

    num_in_test = sum(len(fold) for fold in folds)
    print ("Num Folds: " + str(fold_k) + "\t Num per fold: " + str(num_per_fold))
    print ("Length of Original: " + str(len(train_data)) + "\t Length of Folds: " + str(num_in_test))

    return folds, test_data







def protectedDiv(left, right):
    try:
        if (right < 0.000000000001):
            return 1
        return left / right
    except ZeroDivisionError:
        return 1
    except RuntimeWarning:
        print("Left: " + str(left) + "\tRight: " + str(right))
        return 1

def abs_sqrt(num):
    return math.sqrt(abs(num))

def cbrt(num):
    return sp.special.cbrt(num)

def sin(num):
    try:
    #print("Num: " + str(num))
        n2 = math.radians(num)
    #print("N2: " + str(n2))
        return math.sin(n2)
    except ValueError:
        print("Infinity Warning")
        print("Num: " + str(num))
        #print("N2: " + str(n2))

def cos(num):
    try:
    #print("Num: " + str(num))
        n2 = math.radians(num)
    #print("N2: " + str(n2))
        return math.cos(n2)
    except ValueError:
        print("Infinity Warning")
        print("Num: " + str(num))
        #print("N2: " + str(n2))

def modulo(n1, n2):
    if (n2 == 0):
        return n1
    return n1%n2

def dist(x1,y1,x2,y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

def ifStatement(n1, n2, n3):
    if (n1 < 0):
        return n2
    else:
        return n3

def compStatement(n1, n2, n3, n4):
    if (n1 < n2):
        return n3
    else:
        return n4

def det2x2(n11, n12, n21, n22):
    return (n11*n22-n12*n21)


test_num = 2;
name = "Mut-Experiment-" + str(test_num)





pset = gp.PrimitiveSet("MAIN", 8)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)


if (test_num > 1):
    pset.addPrimitive(sin,1)
#pset.addPrimitive(cos,1)
#pset.addPrimitive(abs_sqrt, 1)
if (test_num > 3):
    pset.addPrimitive(cbrt, 1)
#pset.addPrimitive(max, 2)
#pset.addPrimitive(min, 2)
#pset.addPrimitive(math.floor, 1)
#pset.addPrimitive(math.ceil, 1)
if (test_num > 4):
    pset.addPrimitive(round, 1)
#pset.addPrimitive(modulo, 2)
#pset.addPrimitive(dist,4)
if (test_num > 5):
    pset.addPrimitive(ifStatement,3)
#pset.addPrimitive(compStatement,4)
#pset.addPrimitive(det2x2,4)
if (test_num > 2):
    pset.addPrimitive(math.tanh,1)




pset.addEphemeralConstant("const", lambda: random.uniform(-1, 1))

pset.renameArguments(ARG0="pregnancies")
pset.renameArguments(ARG1="glucose")
pset.renameArguments(ARG2="bPressure")
pset.renameArguments(ARG3="skinThickness")
pset.renameArguments(ARG4="insulin")
pset.renameArguments(ARG5="bmi")
pset.renameArguments(ARG6="pedigreeFunc")
pset.renameArguments(ARG7="age")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)

toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)

def evalFunc(individual, current_k, fold_k, folds):
    func = toolbox.compile(expr=individual)
    total = 1.0

    for i in range(fold_k):
        if (i == current_k):
            continue

#        print ("Folds length: " + str(len(folds)))
        for test in folds[i]:
#            print("Test " + str(test))
#            print("Test " + str(test[:8]))
#            print("Total: " + str(total))
#            print(individual)
#            print("Test Value: " + str(func(*test[:8])))

            total += int((0 if (func(*test[:8])) < 0 else 1) == test[8])


        current_k = (current_k+1)%fold_k
#        print("Total score: " + str(total))
#        total += float(sum(int((func(*(test[:8])) == test[8]) for fold in folds[i] for test in fold)))
#        print ("test: " + str(test[:8]) + "\tRes: " + str(test[8])) for fold in folds[i] for test in fold
    return total, #/(len(train_data)-len(folds[current_k-1]))

def testEval(individual, test_data):
    func = toolbox.compile(expr=individual)
    total = 0
    for test in test_data:
        total += int((0 if (func(*test[:8])) < 0 else 1) == test[8])
    return total/len(test_data)


toolbox.register("evaluate", evalFunc, current_k=current_k, fold_k=fold_k, folds=folds)

toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.register("mate", gp.cxOnePoint)

toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)

toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
stats_size = tools.Statistics(len)

mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
# register methods for calculating various statistics
mstats.register("mean", np.mean)
mstats.register("std", np.std)
mstats.register("min", np.min)
mstats.register("max", np.max)


f_avg = open('Logs/'+name+'-avg.txt', 'w')
avg_vals = []
for i in range(20):
    folds, test_data = splitData(all_data, fold_k)
    toolbox.register("evaluate", evalFunc, current_k=current_k, fold_k=fold_k, folds=folds)

    pop = toolbox.population(n=1000)
    # holds the n best individuals
    hof = tools.HallOfFame(1)
    print ("Run: " + str(i))
    pop, logs = algorithms.eaSimple(pop, toolbox, 0.1, 0.9, 60, stats=mstats, halloffame=hof, verbose=True)
    f = open('Logs/'+name+'-logs-' + str(i) +'.txt', 'w')
    f.write(str(logs))
    f.close()
    #pop2, logs2 = algorithms.eaSimple(pop, toolbox, 0.05, 0.50, 60, stats=mstats, halloffame=hof, verbose=True)
    expr = hof[0]
    print("fitness: " + str(testEval(expr, test_data)))
    avg_vals.append(testEval(expr, test_data))
    #f_avg.write(str(testEval(expr)) + "\n")

f_avg.write("Runs Average: "+str(np.mean(avg_vals))+"\n\n")
f_avg.write(str("\n".join(str(s) for s in avg_vals)))

f_avg.close()
#nodes, edges, labels = gp.graph(expr)

#gpg.graph(nodes, edges, labels)
