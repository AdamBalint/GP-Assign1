
import operator
import math
import random

import numpy

import GP_Graph as gpg

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# safe division catches divide by 0 error
def safe_div(a, b):
    try:
        return a/b
    except ZeroDivisionError:
        return 1

################
# pset constructs the set of terminals and nodes used to build the programs
################

# sets all of the different nodes and terminals
# has the operation first and the number of arguments next
pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(safe_div, 2)
#pset.addPrimitive(math.cos, 1)
#pset.addPrimitive(math.sin, 1)

#When added, a randomly generated constant at the time will be used as the value
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
# rename the input to x for easier reading
pset.renameArguments(ARG0='x')

################
# Creator sets up the main portions of the individual
################

# creates a fitness object with best value -1 since it is minimization
# the weights must be iterable
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Creates rules saying that the individuals are made up of the above defined
# structure and use the fitness function FitnessMin
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


###############
# Toolbox sets up the different methods used in the evolution of the individuals
###############

toolbox = base.Toolbox()
# sets how to build a solution half within the limit, and half naturally
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)

# specifies all components of how to create the individual
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

# create the population as a list by repeating the instructions to create the
# individual using toolbox.individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile the code using the pset as the logic and terminals
toolbox.register("compile", gp.compile, pset=pset)

# defines the symbolic regression to find the error (fitness)
# always returned as an iterable touple
def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - (x**2 + x + 1))**2 for x in points)
    return math.fsum(sqerrors) / len(points),

#sets up the evaluation. evaluates points between -1 and 1 at 0.1 intervals
toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10, 10)])

# sets up the selection, tournament selection with 3 individuals
toolbox.register("select", tools.selTournament, tournsize=3)

# sets up the crossover
toolbox.register("mate", gp.cxOnePoint)

# sets the mutation to generate a full tree of size between 0 and 2
toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)

# sets the mutation
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# sets check on crossover to make sure the height does not go above 17
# done to reduce bloat
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# sets check on mutation to make sure the height does not go above 17
# done to reduce bloat
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

################
# Setting up the statistics
################

# get the fitness of the individuals
stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
# get the number of entries returned?
stats_size = tools.Statistics(len)
# set the values
mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
# register methods for calculating various statistics
mstats.register("mean", numpy.mean)
mstats.register("std", numpy.std)
mstats.register("min", numpy.min)
mstats.register("max", numpy.max)

# create a population of 300
pop = toolbox.population(n=100)
# holds the n best individuals
hof = tools.HallOfFame(1)

# runs the algorithm and keeps the logs. Uses simple version
# Crossover = 0.5, Mutation = 0.1
# nGens = 40
# stats - the stats that will be recorded
# hof - stores the n best individuals
# Verbose - logging statistics True or False
pop, logs = algorithms.eaSimple(pop, toolbox, 0.8, 0.15, 50, stats=mstats, halloffame=hof, verbose=True)

expr = hof[0]

print("fitness" + str(evalSymbReg(expr, points=[x/10. for x in range(-10, 10)])))



nodes, edges, labels = gp.graph(expr)

gpg.graph(nodes, edges, labels)
