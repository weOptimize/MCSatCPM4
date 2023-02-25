#!/home/pinoystat/Documents/python/mymachine/bin/python

#* get execution time 
import time

start_time = time.time()

#get budgetting confidence policy
budgetting_confidence_policies = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#array to store all budgeted durations linked to the budgetting confidence policy
budgeteddurations = []
stdevs = []
#array to store all found solutions
solutions = []
#array to store all results of the monte carlo simulation
mcs_results = []

#*****

import numpy as np
import pandas as pd
import random
import seaborn as sns
from pandas_ods_reader import read_ods
from operator import itemgetter
import matplotlib.pyplot as plt 
from scipy import stats as st
from copulas.multivariate import GaussianMultivariate
from scipy.stats import rv_continuous, rv_histogram, norm, uniform, multivariate_normal, beta
from deap import base, creator, tools, algorithms


from fitter import Fitter, get_common_distributions, get_distributions


#import created scripts:
from task_rnd_triang_with_interrupts_stdev_new_R2 import *
from survival_value_extractor import *


#I define the number of candidates to be considered
nrcandidates = 10

#defining a global array that stores all portfolios generated (and another one for the ones that entail a solution)
tested_portfolios = []
solution_portfolios = []

#defining the correlation matrix to be used in the monte carlo simulation (and as check when the correlations are expected to be 0)
correlation_matrix = []

#defining the function that calculates the net present value of a portfolio of projects
def portfolio_npv(portfolio):
    npv_portfolio = 0
    for i in range(nrcandidates):
        if portfolio[i] == 1:
            npv_portfolio += npv(wacc, cashflows[i])
    return npv_portfolio

#defining the function that stores in an array the net present value of each candidate project
def npvperproject_calculator(wacc, cashflows):
    npvperproject = []
    for i in range(nrcandidates):
        npvperproject.append(npv(wacc, cashflows[i]))
    return npvperproject

#defining the function that calculates the total budget of a portfolio of projects
def portfolio_totalbudget(portfolio):
    totalbudget_portfolio = 0
    for i in range(nrcandidates):
        if portfolio[i] == 1:
            totalbudget_portfolio += bdgtperproject[i]
    return totalbudget_portfolio

# Defining the fitness function
def evaluate(individual):
    total_cost = 0
    total_npv = 0
    for i in range(nrcandidates):
        if individual[i] == 1:
            total_cost += bdgtperproject[i]
            #total_cost += PROJECTS[i][0]
            # add the net present value of the project to the total net present value of the portfolio
            total_npv += npvperproject[i]
            #total_npv += npv[i][1]
    if total_cost > maxbdgt:
        return 0,
    return total_npv,

# Define the genetic algorithm parameters
POPULATION_SIZE = 20
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 100
HALL_OF_FAME_SIZE = 1

# Define the individual creator
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the toolbox
toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, nrcandidates)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Define the hall of fame
hall_of_fame = tools.HallOfFame(HALL_OF_FAME_SIZE)

# Define the statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", max)



# defining the function that maximizes the net present value of a portfolio of projects, while respecting the budget constraint (using a genetic algorithm)
def maximize_npv():
    # Initialize the population
    population = toolbox.population(n=POPULATION_SIZE)
    for generation in range(MAX_GENERATIONS):
        # Vary the population
        offspring = algorithms.varAnd(population, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
        # Update the hall of fame with the generated individuals
        hall_of_fame.update(offspring)
        population = toolbox.select(offspring, k=len(population))    
        record = stats.compile(population)
        print(f"Generation {generation}: Max NPV = {record['max']}")
    # return the optimal portfolio from the hall of fame their fitness and the total budget
    return hall_of_fame[0], hall_of_fame[0].fitness.values[0], portfolio_totalbudget(hall_of_fame[0])

#defining the function that, for each budgetting confidence policy, computes the budgeted duration
#of each project and the standard deviation of the budgeted duration (and the related budgeted cost)
#initialize an array of budgeted durations that is nrcandidates x len(budgetting_confidence_policies)
budgetedcosts = np.zeros((nrcandidates, len(budgetting_confidence_policies)))
#initialize an array of standard deviations that is sized as far as nrcandidates
stdevs = np.zeros((nrcandidates, 1))
for i in range(nrcandidates):
    iterations=100
    #open ten different ODS files and store the results in a list after computing the CPM and MCS
    filename = "RND_Schedules/data_wb" + str(i+1) + ".ods"
    #print(filename)
    mydata = read_ods(filename, "Sheet1")
    #open ten different ODS files and store the results in a list after computing the CPM and MCS
    filename = "RND_Schedules/riskreg_" + str(i+1) + ".ods"
    #print(filename)
    myriskreg = read_ods(filename, "riskreg")
    #compute MonteCarlo Simulation and store the results in an array called "sim_durations"
    sim_costs = MCS_CPM_RR(mydata, myriskreg, iterations)
    #multiply each value in sim_durations by 5000 to get the results in Euros
    #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the iterations
    mcs_results.append(sim_costs)
    for j in range(len(budgetting_confidence_policies)):
        budgetting_confidence_policy = budgetting_confidence_policies[j]
        #print(budgetting_confidence_policy)
        #extract the survival value from the array sim_duration that corresponds to the budgetting confidence policy
        survival_value = survival_value_extractor(sim_costs, budgetting_confidence_policy, iterations)
        #store the first survival value in an array where the columns correspond to the budgetting confidence policies and the rows correspond to the projects
        budgetedcosts[i][j]=survival_value
    #I perform a sumproduct to the array of budgeted durations to get the total budgeted cost (each unit in the array costs 5000 euros) now x1 because I did it before
    #totalbudget=sum(budgetedcosts)*1
    #I multiply each value in the array of budgeted durations by 5000 to get the total budgeted cost per project (each unit in the array costs 5000 euros) keeping the same type of array
    bdgtperproject_matrix=budgetedcosts*1

#check the parameters of beta distribution for each of the mcs_results
betaparams = []
for i in range(nrcandidates):
    f = Fitter(mcs_results[i], distributions=['beta'])
    f.fit()
    betaparam=(f.fitted_param["beta"])
    betaparams.append(betaparam)

#extract all "a" parameters from the betaparams array
a = []
for i in range(nrcandidates):
    a.append(betaparams[i][0])

#extract all "b" parameters from the betaparams array
b = []
for i in range(nrcandidates):
    b.append(betaparams[i][1])

#extract all "loc" parameters from the betaparams array
loc = []
for i in range(nrcandidates):
    loc.append(betaparams[i][2])

#extract all "scale" parameters from the betaparams array
scale = []
for i in range(nrcandidates):
    scale.append(betaparams[i][3])


print(betaparams)


# copy the array with all MCS results
df0 = pd.DataFrame(data=mcs_results).T
df0.rename(columns={0:"P01", 1:"P02", 2:"P03", 3:"P04", 4:"P05", 5:"P06", 6:"P07", 7:"P08", 8:"P09", 9:"P10"}, inplace=True)
correlation_matrix0 = df0.corr()

# this function calculates the npv of each project and then uses the maximizer function to obtain and return portfolio, npv and bdgt in a matrix (solutions)
for i in range(len(budgetting_confidence_policies)):
    #I take the column of bdgtperproject_matrix that corresponds to the budgetting confidence policy
    bdgtperproject=bdgtperproject_matrix[:,i]
    #I define the budget constraint #was 250k
    maxbdgt = 3800
    #open a file named "expected_cash_flows.txt", that includes ten rows and five columns, and store the values in a list. Each row corresponds to a project, and each column corresponds to a year
    cashflows = []
    with open('RND_Schedules/expected_cash_flows.txt') as f:
        j=0
        for line in f:
            cashflows.append([float(x) for x in line.split()])
            #substract the budgeted cost (inside bdgtperproject) from the first column (year 0) of the cashflows for each corresponding project
            cashflows[j][0] = cashflows[j][0] - bdgtperproject[j]
            #cashflows[j][0] = cashflows[j][0]
            j=j+1
    #initialize a variable that reflects the weighted average cost of capital
    wacc = 0.1
    #defining the function that calculates the net present value of a project
    def npv(rate, cashflows):
        return sum([cf / (1 + rate) ** k for k, cf in enumerate(cashflows)])
    # call the function that calculates the npv of each candidate project
    npvperproject = npvperproject_calculator(wacc, cashflows)

    projectselection = maximize_npv()
    #assign the result from projectselection to the variable solutions
    solutions.append(projectselection)
    #print(solutions)

#separate the npv results from the solutions list
npv_results = [round(x[1], 0) for x in solutions]
#separate the portfolio results from the solutions list
portfolio_results = [x[0] for x in solutions]
#separate the budgets taken from the solutions list
budgets = [x[2] for x in solutions]

#DESACTIVAR ALL THIS SI QUIERES MIRAR TODOS JUNTOS - HASTA PLT(SHOW)
plt.figure(1)
plt.scatter(budgetting_confidence_policies, npv_results, color='grey')
#zoom in the plot so that the minumum value of the x axis is 0.5 and the maximum value of the x axis is 1
plt.title("NPV vs Budgetting Confidence Policy")
plt.xlabel("Budgetting Confidence Policy")
plt.ylabel("NPV")
#add the values of the npv results to the plot as annotations and displaced vertically a 1% of the y axis
for i, txt in enumerate(npv_results):
    txt = "{:,}".format(round(txt))
    plt.annotate(txt, (budgetting_confidence_policies[i], npv_results[i]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.xlim(0.45, 1)
#increase the size of all the fonts in the plot
plt.rcParams.update({'font.size': 16})
plt.grid()		
#plt.show()



plt.show()
#*** execution time
print("Execution time: %s milli-seconds" %((time.time() - start_time)* 1000))


  



