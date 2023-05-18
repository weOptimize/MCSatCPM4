#This file includes the function that returns the survival value for a given budgetting confidence policy.
#It is called from the main file
import math
import random
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from operator import itemgetter
from pandas_ods_reader import read_ods


#import created scripts:
#from task_rnd_triang_with_interrupts_stdev_new_R2 import *
#from simulate_function_mcs import *

#I define the number of candidates to be considered
nrcandidates = 10
nr_confidence_policies = 1
from old.RiskReg_bdgt_vs_policies_PFsim_stochNPV import maxbdgt, df10r, iterations

#initialize matrices to store bdgt and npv
bdgtperproject_matrix = np.zeros((nrcandidates, nr_confidence_policies))
npvperproject_matrix = np.zeros((nrcandidates, nr_confidence_policies))

# Defining the fitness function
def evaluate(individual, bdgtperproject, npvperproject, maxbdgt, df10r, iterations):
    #multiply dataframe 10r by the chosen portfolio to reflect the effect of the projects that are chosen
    pf_df10r = df10r * individual
    #sum the rows of the new dataframe to calculate the total cost of the portfolio
    pf_cost10r = pf_df10r.sum(axis=1)

    #extract the maximum of the resulting costs
    maxcost10r = max(pf_cost10r)
    print("max cost:")
    print(maxcost10r)
    #count how many results were higher than maxbdgt
    count = 0
    for i in range(pf_cost10r.__len__()):
        if pf_cost10r[i] > maxbdgt:
            count = count + 1
    #array storing the portfolio risk not to exceed 3.800 Mio.€, as per-one risk units
    portfolio_risk = 1-count/iterations
    total_cost = 0
    total_npv = 0
    for i in range(nrcandidates):
        #print(total_cost)
        if individual[i] == 1:
            total_cost += bdgtperproject[i]
            #total_cost += PROJECTS[i][0]
            # add the net present value of the project to the total net present value of the portfolio
            total_npv += npvperproject[i]
            #total_npv += npv[i][1]
    if total_cost > maxbdgt: #or portfolio_risk > 0.8:
        return 0,
    return total_npv


# Define the genetic algorithm parameters
POPULATION_SIZE = 50 #was 100
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 200 #was 500
HALL_OF_FAME_SIZE = 1

# Create the individual and population classes based on the list of attributes and the fitness function
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# create the Individual class based on list
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the toolbox
toolbox = base.Toolbox()
# register a function to generate random integers (0 or 1) for each attribute/gene in an individual
toolbox.register("attr_bool", random.randint, 0, 1)
# register a function to generate individuals (which are lists of several -nrcandidates- 0s and 1s -genes-
# that represent the projects to be included in the portfolio)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, nrcandidates)
# register a function to generate a population (a list of individuals -candidate portfolios-)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# register the goal / fitness function
toolbox.register("evaluate", evaluate, bdgtperproject=bdgtperproject_matrix, npvperproject=npvperproject_matrix, maxbdgt=maxbdgt, df10r=df10r, iterations=iterations)
# register the crossover operator (cxTwoPoint) with a probability of 0.9 (defined above)
toolbox.register("mate", tools.cxTwoPoint)
# register a mutation operator with a probability to flip each attribute/gene of 0.05.
# indpb is the independent probability for each gene to be flipped and P_MUTATION is the probability of mutating an individual
# The difference between P_MUTATION and indpb is that P_MUTATION determines whether an individual will be mutated or not,
# while indpb determines how much an individual will be mutated if it is selected for mutation.
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the hall of fame
hall_of_fame = tools.HallOfFame(HALL_OF_FAME_SIZE)

# Define the statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", max)

# defining the function that maximizes the net present value of a portfolio of projects, while respecting the budget constraint (using a genetic algorithm)
def maximize_npv():
    # Empty the hall of fame
    hall_of_fame.clear()
    print("****************new policy iteration****************")
    # Initialize the population
    population = toolbox.population(n=POPULATION_SIZE)
    for generation in range(MAX_GENERATIONS):
        # Vary the population
        offspring = algorithms.varAnd(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION)
        # Evaluate the new individuals fitnesses
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        # Update the hall of fame with the generated individuals
        hall_of_fame.update(offspring)
        # reorder the hall of fame so that the highest fitness individual is first
        # hall_of_fame.sort(key=itemgetter(0), reverse=True)
        population = toolbox.select(offspring, k=len(population))    
        record = stats.compile(population)
        print(f"Generation {generation}: Max NPV = {record['max']}")

    #de momento me dejo de complicarme con el hall of fame y me quedo con el último individuo de la última generación
    # return the optimal portfolio from the hall of fame, their fitness and the total budget
    #print(hall_of_fame[0], hall_of_fame[0].fitness.values[0], portfolio_totalbudget(hall_of_fame[0], bdgtperproject_matrix))
    return hall_of_fame[0], hall_of_fame[0].fitness.values[0], portfolio_totalbudget(hall_of_fame[0], bdgtperproject_matrix)
    #print(hall_of_fame)
    #return hall_of_fame

#defining the function that calculates the total budget of a portfolio of projects
def portfolio_totalbudget(portfolio,bdgtperproject):
    totalbudget_portfolio = 0
    #totalbudget_npv = 0
    for i in range(nrcandidates):
        if portfolio[i] == 1:
            totalbudget_portfolio += bdgtperproject[i]
            #totalbudget_npv += npvperproject[i]
    #return totalbudget_portfolio, totalbudget_npv
    return totalbudget_portfolio

    





