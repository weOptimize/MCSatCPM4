#!/home/pinoystat/Documents/python/mymachine/bin/python

#* get execution time 
import time
import numpy as np
import pandas as pd
import seaborn as sns
from pandas_ods_reader import read_ods
from operator import itemgetter
import matplotlib.pyplot as plt 
from scipy import stats as st
from deap import base, creator, tools, algorithms
#from copulas.multivariate import GaussianMultivariate
#from scipy.stats import rv_continuous, rv_histogram, norm, uniform, multivariate_normal, beta
#from fitter import Fitter, get_common_distributions, get_distributions
import seaborn as sns
import sys

#import created scripts:
from task_rnd_triang_with_interrupts_stdev_new_R2 import *
from functions_for_simheuristic_v12 import *

# import Threshold_calculation_vs05
import Threshold_calculation_other_DETERMINISTICs


# create an empty list to store the timestamps and labels
timestamps = []

# Redirect optput to a file instead of Terminal
# Save current standard output
stdout = sys.stdout

# Redirect standard output to a file
sys.stdout = open('output.txt', 'w')


start_time = time.time()
timestamps.append(('t = 0', time.time()))

#get budgetting confidence policy
#budgetting_confidence_policies = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
budgetting_confidence_policies = [0.75]
#budgetting_confidence_policies = [0.95]
#array to store all budgeted durations linked to the budgetting confidence policy
budgeteddurations = []
stdevs = []
#array to store all found solutions
solutions = []
#arrays to store all results of the monte carlo simulation
mcs_results = []
mcs_results1 = []
mcs_results2 = []
#defining a global array that stores all portfolios generated (and another one for the ones that entail a solution)
tested_portfolios = []
solution_portfolios = []
PV_results = []
budgets = []

#defining the correlation matrix to be used in the monte carlo simulation (and as check when the correlations are expected to be 0)
correlation_matrix = []


#*****


#I define the number of candidates to be considered and the number of iterations for the MCS
nrcandidates = 20
iterations = 50 #was 300 #was 500
iterations_finalMCS = 500 #was 5k
iterations_postpro = 100

#iterations = 30
#iterations_finalMCS = 50

#I define the budget constraint (in k€) and the minimum confidence level for the portfolio
maxbdgt = 10800
min_pf_conf = 0.90

#initialize an array of budgeted durations that is nrcandidates x len(budgetting_confidence_policies)
budgetedcosts = np.zeros((nrcandidates, len(budgetting_confidence_policies)))

#I define a candidate array of size nr candidates with all ones
candidatearray = np.ones(nrcandidates)
#I define an initial array of indexes with all candidates ranging from 0 to nrcandidates-1
initial_projection_indexes = np.arange(nrcandidates)

#first simulation to get all cdfs for cost & benefits before optimization step (may_update: was 1000)
mcs_results1 = simulate(candidatearray,iterations)

#print("mcs results1: ", mcs_results1[0])

# mcs_results1[0] corresponds to the project costs and mcs_results1[1] to the project benefits (PV)
x_perproj_matrix1 = pointestimate(mcs_results1[0], mcs_results1[1], budgetting_confidence_policies, nrcandidates)
print ("x_perproj_matrix1: ", x_perproj_matrix1)

# write the first timestamp and label to the list
timestamps.append(('First MCS with point estimate of budgets and PV for each project', time.time()))

# extract first column of the matrix to get the budgeted costs of each project and store it in bdgtperproject_matrix
bdgtperproject_matrix = x_perproj_matrix1[0]
# extract second column of the matrix to get the PV of each project and store it in PVperproject_matrix
PVperproject_matrix = x_perproj_matrix1[1]
# print("bdgtperproject_matrix at MAIN: ", bdgtperproject_matrix)
# print("PVperproject_matrix at MAIN: ", PVperproject_matrix)
# print("x_perproj_matrix1: ", x_perproj_matrix1)
# sum the costs of all projects to get the total cost of the portfolio if choosing all projects
totalcost = np.sum(x_perproj_matrix1[0])

# perform a boxplot of the costs obtained in the montecarlo simulation (mcs_results1[0]),
# one boxplot per project (as many projects as nrcandidates) with the corresponding label
plt.figure()
# title: cost distributions of each candidate project in stochastic scenario
plt.boxplot(mcs_results1[0], labels=range(1,nrcandidates+1))
# set min and max for y-axis
plt.ylim(0, 5000)
plt.title("Cost distributions of each candidate project in stochastic scenario")

# create a list that results by substracting MCS_benefits (- MCS_costs)
PV_results = [0] * len(mcs_results1[0])
for i in range(len(mcs_results1[0])):
#    PV_results[i] = np.array(mcs_results1[1][i]) - np.array(mcs_results1[0][i])
    PV_results[i] = np.array(mcs_results1[1][i])

#create new plot
plt.figure()
# title: PV distributions of each candidate project in stochastic scenario
plt.boxplot(PV_results, labels=range(1,nrcandidates+1))
# set min and max for y-axis
plt.ylim(0, 6000)
plt.title("PV distributions of each candidate project in stochastic scenario")
# plt.show()


# print("total portfolio cost allocation request (without correlations because it is a request):")
# print(totalcost)

df10r = correlatedMCS(mcs_results1, iterations, nrcandidates, initial_projection_indexes)
# print("df10r: ", df10r)

# write the second timestamp (substract the current time minus the previously stored timestamp) and label to the list
timestamps.append(('First MCS with correlated cost and PV for each project', time.time()))

# Defining the fitness function
def evaluate(individual, bdgtperproject, PVperproject, maxbdgt):
    total_cost = 0
    total_PV = 0
    #multiply dataframe 10r by the chosen portfolio to reflect the effect of the projects that are chosen
    pf_df10r = df10r * individual
    #sum the rows of the new dataframe to calculate the total cost of the portfolio
    pf_cost10r = pf_df10r.sum(axis=1)
    #extract the maximum of the resulting costs
    maxcost10r = max(pf_cost10r)
    #print("max cost:")
    #print(maxcost10r)
    #count how many results were higher than maxbdgt
    count = 0
    for i in range(pf_cost10r.__len__()): #this is repeated as much as values are in the MCS results for project cost
        if pf_cost10r[i] > maxbdgt:
            count = count + 1
    #array storing the portfolio risk not to exceed 10.800 Mio.€, as per-one risk units
    portfolio_confidence = 1-count/iterations
    #print("portfolio confidence:")
    #print(portfolio_confidence)
    for i in range(nrcandidates):
        #print(total_cost)
        if individual[i] == 1:
            total_cost += bdgtperproject[i]
            #total_cost += PROJECTS[i][0]
            # add the net present value of the project to the total net present value of the portfolio
            total_PV += PVperproject[i]
            #total_PV += PV[i][1]
    # if total_cost > maxbdgt or portfolio_confidence < min_pf_conf: 
    if portfolio_confidence < min_pf_conf: #"total_cost > maxbdgt" removed to prevent affectation of point estimate
        return 0, 0
    return total_PV, portfolio_confidence

# Define the genetic algorithm parameters
POPULATION_SIZE = 180 #was 100 #was 50 #was 180/30
#POPULATION_SIZE = 40
P_CROSSOVER = 0.4
P_MUTATION = 0.6
MAX_GENERATIONS = 400 #was 500 #was 200 #was 100 #was 300 
#MAX_GENERATIONS = 100
HALL_OF_FAME_SIZE = 5

# Create the individual and population classes based on the list of attributes and the fitness function # was weights=(1.0,) returning only one var at fitness function
creator.create("FitnessMax", base.Fitness, weights=(100000.0, 1.0))
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
toolbox.register("evaluate", evaluate, bdgtperproject=bdgtperproject_matrix, PVperproject=PVperproject_matrix, maxbdgt=maxbdgt)
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

# defining the function that maximizes the present value (PV) of a portfolio of projects, while respecting the budget constraint (using a genetic algorithm)
def maximize_PV():
    # Empty the hall of fame
    hall_of_fame.clear()
    # print("****************new policy iteration****************")
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
        print(f"Generation {generation}: Max PV = {record['max']}")

    #de momento me dejo de complicarme con el hall of fame y me quedo con el último individuo de la última generación
    # return the optimal portfolio from the hall of fame, their fitness and the total budget
    # print(hall_of_fame)
    #return hall_of_fame
    print("Hall of Fame:")
    for i in range(HALL_OF_FAME_SIZE):
        print(hall_of_fame[i], hall_of_fame[i].fitness.values[0], hall_of_fame[i].fitness.values[1], portfolio_totalbudget(hall_of_fame[i], bdgtperproject_matrix))
    #print(hall_of_fame[0], hall_of_fame[0].fitness.values[0], hall_of_fame[0].fitness.values[1], portfolio_totalbudget(hall_of_fame[0], bdgtperproject_matrix))
    #print(hall_of_fame[1], hall_of_fame[1].fitness.values[0], hall_of_fame[1].fitness.values[1], portfolio_totalbudget(hall_of_fame[1], bdgtperproject_matrix))
    #print(hall_of_fame[2], hall_of_fame[2].fitness.values[0], hall_of_fame[2].fitness.values[1], portfolio_totalbudget(hall_of_fame[2], bdgtperproject_matrix))
    #return hall_of_fame[0], hall_of_fame[0].fitness.values[0][0], portfolio_totalbudget(hall_of_fame[0], bdgtperproject_matrix)
    return hall_of_fame

# this function calculates the PV of each project and then uses the maximizer function to obtain and return portfolio, PV and bdgt in a matrix (solutions)
for i in range(len(budgetting_confidence_policies)):
    # I take the column of bdgtperproject_matrix that corresponds to the budgetting confidence policy
    bdgtperproject=bdgtperproject_matrix[:,i]
    # print(bdgtperproject)
    PVperproject=PVperproject_matrix[:,i]
    # print(PVperproject)
    # execute the maximizer function to obtain the portfolio, and its PV and bdgt
    projectselection = maximize_PV()
    # assign the result from projectselection to the variable solutions
    solutions.append(projectselection)
    #print(solutions)
# lately I only had one BCP, si it has performed the append only once, however as the solution is a hall of fame, it has appended a list of 3 individuals

#store the PV results, portfolio results, portfolio confidence levels and budgets taken in different lists
PV_results = [0] * len(projectselection)
portfolio_results = [0] * len(projectselection)
portfolio_confidence_levels = [0] * len(projectselection)
pf_conf2 = [0] * len(projectselection)
budgets = [0] * len(projectselection)
for i in range(nrcandidates):
    PV_results = [[x[i].fitness.values[0][0] for x in solutions] for i in range(len(projectselection))]
    #portfolio_results = [[x[i] for x in solutions] for i in range(len(projectselection))]
    portfolio_confidence_levels = [[x[i].fitness.values[1] for x in solutions] for i in range(len(projectselection))]
    budgets = [[portfolio_totalbudget(x[i], bdgtperproject_matrix)[0] for x in solutions] for i in range(len(projectselection))]

# take all arrays inside portfolio_results and sum all of them
portfolio_projection = [sum(x) for x in zip(*projectselection)]

# convert portfolio_projection array into a binary array, where 1 means that the project is selected and 0 means that it is not
portfolio_projection = [1 if x > 0 else 0 for x in portfolio_projection]

# calculate the amount of projects in "portfolio_projection"
projected_candidates = sum(portfolio_projection)

# store the positions of the chosen projects in the portfolio_projection array, starting with 0 (as i+1 for if if starting with 1)
zipped_projection_indexes = [i for i, x in enumerate(portfolio_projection) if x == 1]

# convert portfolio_projection in a full ones array
# portfolio_projection = [1] * len(portfolio_projection)

# write the third timestamp (substract the current time minus the previously stored timestamp) and label to the list
timestamps.append(('Optimization step (GA algorithm)', time.time()))


print ("************ SUMMARY STAGE 1 **********")
print ("PV_results: ", PV_results)
print ("portfolio_results: ", projectselection)
print ("portfolio_confidence_levels: ", portfolio_confidence_levels)
print ("budgets: ", budgets)
print ("portfolio_projection: ", portfolio_projection)
print ("Indexes of selected projects: ", zipped_projection_indexes)
print ("Number of candidate projects for stage 2: ", projected_candidates)

print ("************ STARTING STAGE 2 (long MCS) **********")
#second simulation to get all cdfs for cost & benefits after optimization step (may_update: was 1000)
mcs_results2 = simulate(portfolio_projection,iterations_finalMCS) # we pass portfolio_projection to simulate only the selected projects
# we obtain a matrix sized (iterations_finalMCS, 2) where mcs_results2[0] corresponds to the project costs
# and mcs_results2[1] to the project benefits (PV) each of them with "iterations_finalMCS" rows

# mcs_results2[0] corresponds to the project costs and mcs_results2[1] to the project benefits (PV_SINdescontarCOSTE)
x_perproj_matrix2 = pointestimate(mcs_results2[0], mcs_results2[1], budgetting_confidence_policies, projected_candidates)
# print ("x_perproj_matrix2: ", x_perproj_matrix2)

# write the fourth timestamp and label to the list
timestamps.append(('Second MCS, also including point estimate of budgets and PV for shortlisted projects', time.time()))

# we assume correlations at the cost side, not at the benefits side (conservative approach)
# update x_perproj_matrix2 with the correlation effect registered inside df20r
# print("x_perproj_matrix2: ", x_perproj_matrix2)
# separate the budget and PV results from the x_perproj_matrix
bdgtperproject_matrix = x_perproj_matrix2[0]
PVperproject_matrix = x_perproj_matrix2[1]
# print(type(bdgtperproject_matrix))
# print(type(PVperproject_matrix))
bdgtperproject_matrix = np.squeeze(bdgtperproject_matrix)
PVperproject_matrix = np.squeeze(PVperproject_matrix)

# remove all data that has zeroes from bdgtperproject_matrix and PVperproject_matrix
# bdgtperproject_matrix = bdgtperproject_matrix[np.nonzero(bdgtperproject_matrix.flatten())]
# PVperproject_matrix = PVperproject_matrix[np.nonzero(PVperproject_matrix.flatten())]

# print("bdgtperproject_matrix: ", bdgtperproject_matrix)
# print("PVperproject_matrix: ", PVperproject_matrix)
print("size of bdgtperproject_matrix", len(bdgtperproject_matrix))
print("size of PVperproject_matrix", len(PVperproject_matrix))
print("size of mcs_results2", len(mcs_results2))

# print("mcs_results2 (input para correlacionar): ", mcs_results2)

# for each of the options obtained in projectselection, calculate the total portfolio PV and the portfolio budget based on the information
# from x_perproj_matrix
PV_results = [0] * len(projectselection) # as many as len(projectselection) because we have one PV per item in HoF
budgets = [0] * len(projectselection)
pf_conf2 = [0] * len(projectselection)
widened_bdgtperproject_matrix = [0] * nrcandidates # as many as initial amount of project candidates
widened_PVperproject_matrix = [0] * nrcandidates
# initialize dataframe called widened_df20r as a copy of df10r
widened_df20r = df10r.copy()
# enlarge the dataframe to the size of iterations_finalMCS
widened_df20r = widened_df20r.reindex(range(iterations_finalMCS))
# fill the dataframe with zeroes
widened_df20r.iloc[:, :] = 0

df20r = correlatedMCS(mcs_results2, iterations_finalMCS, projected_candidates, zipped_projection_indexes)
#we obtain a matrix sized (iterations_finalMCS, 2) where df20r[0] corresponds to the project costs
# and df20r[1] to the project benefits (PV) each of them with "iterations_finalMCS" rows


# pick in order the values from bdgtperproject_matrix and PVperproject_matrix and store them in widened_bdgtperproject_matrix
# and widened_PVperproject_matrix
# The location of the values to be picked is available in zipped_projection_indexes
j=0
for i in range(nrcandidates):
    if i in zipped_projection_indexes:
        widened_bdgtperproject_matrix [i] = round(bdgtperproject_matrix [j],3)
        widened_PVperproject_matrix [i] = round(PVperproject_matrix [j],3)
        j+=1
    else:
        pass
# print("widened_bdgtperproject_matrix: ", widened_bdgtperproject_matrix)
# print("widened_PVperproject_matrix: ", widened_PVperproject_matrix)


# pick in order the values from df20r and store them in widened_df20r (to be used in the next step)
i=0
j=0
k=0
for i in range(nrcandidates):
    if i in zipped_projection_indexes:
        for j in range(iterations_finalMCS):
            widened_df20r.loc[j, widened_df20r.columns[i]] = df20r.loc[j, df20r.columns[k]]
        k += 1
    else:
        pass

print("widened_df20r: ", widened_df20r)

for i in range(len(projectselection)):
    #calculate the total portfolio budget by multiplying the budget of each project by the binary array obtained in projectselection    
    print(projectselection[i])
    budgets[i] = np.sum(np.multiply(widened_bdgtperproject_matrix,projectselection[i]))
    #calculate the total portfolio PV by multiplying the PV of each project by the binary array obtained in projectselection
    PV_results[i] = np.sum(np.multiply(widened_PVperproject_matrix,projectselection[i]))
    #multiply dataframe 20r by the chosen portfolio to reflect the effect of the projects that are chosen
    pf_df20r = widened_df20r * projectselection[i]
    #sum the rows of the new dataframe to calculate the total cost of the portfolio
    pf_cost20r = pf_df20r.sum(axis=1)
    #extract the maximum of the resulting costs
    maxcost20r = max(pf_cost20r)
    print("max cost:")
    print(maxcost20r)
    #count how many results were higher than maxbdgt
    count = 0
    for j in range(pf_cost20r.__len__()):
        if pf_cost20r[j] > maxbdgt:
            count = count + 1
    #array storing the portfolio risk not to exceed 3.800 Mio.€, as per-one risk units
    pf_conf2[i] = 1-count/iterations_finalMCS

# create a dataframe with the results
finalsol_df = pd.DataFrame({'Portfolio': projectselection, 'Portfolio PV': PV_results, 'Portfolio Budget': budgets, 'Portfolio confidence': pf_conf2})
# order the dataframe by the portfolio PV, starting with the highest PV
finalsol_df = finalsol_df.sort_values(by=['Portfolio PV'], ascending=False)
#sum the amount of projects that have been chosen (binary value = 1)
finalsol_df['Portfolio size'] = finalsol_df['Portfolio'].apply(lambda x: sum(x))
print ("Final Solution: ", finalsol_df)
#extract the amount of projects chosen in the best portfolio
bestsol_size = finalsol_df.iloc[0,4] # deleted "+1" to check same amount of projects like optimal

# write the fifth timestamp and label to the list. Calculation FINALIZED
timestamps.append(('Application of correlation effect to final options', time.time()))

segments = [0] * (len(timestamps)-1)

# calculate the difference between each pair of timestamps
for i in range(0, len(timestamps)-1):
    segments[i] = (timestamps[i+1][0], round(timestamps[i+1][1] - timestamps[i][1], 2))
    print(segments)
    
# create a dataframe from the list of timestamps
crono_frame = pd.DataFrame(segments, columns=['Checkpoint', 'Execution time (s)'])

# add a final register with the total execution time
crono_frame.loc['Total'] = ['Total', crono_frame['Execution time (s)'].sum()]

# print the dataframe
print(crono_frame)

PV_results = []
budgets = []
pf_cost20r = []
#pf_conf2 = []

#from the sorted dataframe, take the first row, which corresponds to the highest PV portfolio and extract the data needed for the following pictures
finalsol_df = finalsol_df.iloc[0]
best_stoch_pf = finalsol_df[0]
PV_results_escalar = finalsol_df[1]
PV_results.append(PV_results_escalar)
#PV_results.append(finalsol_df[1])
budgets_escalar = finalsol_df[2]
budgets.append(budgets_escalar)
#budgets.append(finalsol_df[2])
print("portfolio_results: ", best_stoch_pf)
print("PV_results: ", PV_results)
print("budgets: ", budgets)


#*** Total execution time
print("Total execution time: %s seconds" %((time.time() - start_time)))

# execute the code inside Threshold_calculation vs05.py to check the thresholds for
# checking algorithm plausibility and extract the two deterministic portfolios obtained
deterministic_portfolio_with_reserv, deterministic_portfolio_2kshift = Threshold_calculation_other_DETERMINISTICs.deterministic_with_reserves(df10r, bestsol_size)


# I want to reuse code that analyzes a HoF, but now I want it to analyze only one solution, 
# so I create a list with only one element
projectselection = []
projectselection.append(deterministic_portfolio_with_reserv)# simulate the portfolio to obtain the MCS of PV and portfolio costs and then graph the results

projectselection = []
projectselection.append(deterministic_portfolio_2kshift)# simulate the portfolio to obtain the MCS of PV and portfolio costs and then graph the results


# simulatescenario(df10r, portfolio_projection, projectselection, iter):
mcs_results4, widened_df20r4 = simulatescenario(df10r, deterministic_portfolio_with_reserv, projectselection, iterations_postpro)

# simulatescenario(df10r, portfolio_projection, projectselection, iter):
mcs_results5, widened_df20r5 = simulatescenario(df10r, deterministic_portfolio_2kshift, projectselection, iterations_postpro)

# store the positions of the chosen projects in the portfolio_projection array, starting with 0 (as i+1 for if if starting with 1)
zipped_indexes_4 = [i for i, x in enumerate(deterministic_portfolio_with_reserv) if x == 1]
zipped_indexes_5 = [i for i, x in enumerate(deterministic_portfolio_2kshift) if x == 1]


# ******************************** POSTPROCESSING ********************************


# mcs_results2 is a three dimensional list of lists sized 2x20xlen(df20r). Extract the two two-dimensional matrices that corresponds
# to the second list of the first dimension
# mcs results 3 has the missing MCS results from
mcs_results2_PVs = mcs_results2[1] 
mcs_results4_PVs = mcs_results4[1] # PVS do not discount project cost yet
mcs_results5_PVs = mcs_results5[1] 
# transpose the list of lists
mcs_results2_PVs = np.transpose(mcs_results2_PVs)
mcs_results4_PVs = np.transpose(mcs_results4_PVs)
mcs_results5_PVs = np.transpose(mcs_results5_PVs)

# create a numpy array and is initialized with the value 0. The resulting matrix will have
# the dimensions iterations_MCS*nrcandidates
nparray_mcs_results2_PVs = np.zeros((iterations_finalMCS, nrcandidates))
nparray_mcs_results4_PVs = np.zeros((iterations_postpro, nrcandidates))
nparray_mcs_results5_PVs = np.zeros((iterations_postpro, nrcandidates))

# take the indexes from zipped_projection_indexes
# and use them to fill the nparray_mcs_results2_PVs with the PVs of the chosen projects
# do it by copying the values of the ith column of mcs_results2_PVs into the jth column
# of nparray_mcs_results2_PVs, being j the value of the ith element of zipped_projection_indexes
j=0
for i in range(iterations_finalMCS):
    if i in zipped_projection_indexes:
        nparray_mcs_results2_PVs [:,i] = mcs_results2_PVs [:,j]
        j+=1
    else:
        pass

j=0
for i in range(nrcandidates):
    if i in zipped_indexes_4:
        nparray_mcs_results4_PVs [:,i] = mcs_results4_PVs [:,j]
        j+=1
    else:
        pass

j=0
for i in range(nrcandidates):
    if i in zipped_indexes_5:
        nparray_mcs_results5_PVs [:,i] = mcs_results5_PVs [:,j]
        j+=1
    else:
        pass


# print nparray_mcs_results2_PVs
# convert nparray_mcs_results2_PVs to a numpy array
# print ("nparray_mcs_results2_PVs: ", nparray_mcs_results2_PVs)
# print ("nparray_mcs_results4_PVs: ", nparray_mcs_results4_PVs)
# print ("nparray_mcs_results5_PVs: ", nparray_mcs_results5_PVs)

# so that only the chosen optimal projects are considered for the next calculations
# nparray_mcs_results2_PVs = np.array(nparray_mcs_results2[1, :, :]) # PVS do not discount project cost yet
# multiply matrix by best_stoch_pf so that only the chosen projects are considered
nparray_mcs_results2_PVs_onlyChosen_pf = nparray_mcs_results2_PVs * best_stoch_pf
nparray_mcs_results_4_PVs = nparray_mcs_results4_PVs * deterministic_portfolio_with_reserv
nparray_mcs_results_5_PVs = nparray_mcs_results5_PVs * deterministic_portfolio_2kshift

# convert df20r to a numpy array, and multiply it by the best portfolio found
# so that only the chosen optimal projects are considered for the next calculations
nparray_df20r = np.array(widened_df20r)
nparray_df20r4 = np.array(widened_df20r4)
nparray_df20r5 = np.array(widened_df20r5)
# multiply matrix by best_stoch_pf so that only the chosen projects are considered
nparray_df20r_onlyChosen_pf = nparray_df20r * best_stoch_pf
nparray_df20r_4 = nparray_df20r4 * deterministic_portfolio_with_reserv
nparray_df20r_5 = nparray_df20r5 * deterministic_portfolio_2kshift

# create new array sized the same as df20r to store the net PV's
netPV = np.zeros(iterations_finalMCS)
netPV_4 = np.zeros(iterations_postpro)
netPV_5 = np.zeros(iterations_postpro)

#initialize array to store the correlated costs of the portfolio
correlated_pfCosts = np.zeros(iterations_finalMCS)
pfCosts_4 = np.zeros(iterations_postpro)
pfCosts_5 = np.zeros(iterations_postpro)

# calculate the net PV by substracting PV minus all costs
for i in range(iterations_finalMCS):
    netPV[i] = nparray_mcs_results2_PVs_onlyChosen_pf[i,:].sum()
    correlated_pfCosts [i] = nparray_df20r_onlyChosen_pf[i,:].sum()

# calculate the net PV by substracting PV minus all costs
for i in range(iterations_postpro):
#    netPV_4[i] = nparray_mcs_results_4_PVs[i,:].sum()- nparray_df20r_4[i,:].sum()
#    netPV_5[i] = nparray_mcs_results_5_PVs[i,:].sum()- nparray_df20r_5[i,:].sum()
    netPV_4[i] = nparray_mcs_results_4_PVs[i,:].sum()
    netPV_5[i] = nparray_mcs_results_5_PVs[i,:].sum()
    pfCosts_4 [i] = nparray_df20r_4[i,:].sum()
    pfCosts_5 [i] = nparray_df20r_5[i,:].sum()

# Boxplot of the correlated montecarlo results of the PV of the portfolio obtained with limit 0.9 confidence
# and compare it respect to the threshold results of the deterministic model
plt.figure(5)
# plt.boxplot([pf_cost, pf_cost10r], labels=['Independent', 'Correlated'])
plt.boxplot([netPV, netPV_4, netPV_5], labels=
            ['PV', 'PV_Deterministic_CR', 'PV_Deterministic -2.5k€', ]) #add ,vert=False if you want it horizontal
plt.title("PV of Optimal portf. vs. Deterministic pf. in stoch.env. vs. lowering Bdgt Limit -2.5k€")
plt.figure(6)
plt.boxplot([correlated_pfCosts, pfCosts_4, pfCosts_5], labels=
            ['Portfolio Costs', 'Portfolio Costs CR', 'Portf.Costs -2.5k€']) #add ,vert=False if you want it horizontal
plt.title("COSTS of Optimal portf. vs. Deterministic pf. in stoch.env. vs. lowering Bdgt Limit -2.5k€")
#set y axis limits between 8000 and 20000
#plt.ylim(7700,37000)
#show the plot
plt.show()
# print netPV and correlated_pfCosts, including their labels, in the same line
# print('Net PV: ', netPV)
# print('Correlated Portfolio Costs: ', correlated_pfCosts)
# show average value of netPV
print('Average PV: ', netPV_4.mean())
# provide value of 90% confidence interval of correlated_pfCosts
print('90% fulfilment confidence of portfolio costs (Det/Conting.Reserves): ', np.percentile(pfCosts_4, [0, 90]))
print('90% fulfilment confidence of portfolio costs (Det/ -2.5 kEur): ', np.percentile(pfCosts_5, [0, 90]))
# estimate the confidence value that corresponds to pfCosts = 10800
sorted_arr = np.sort(pfCosts_4)
index = np.searchsorted(sorted_arr, 10800)
percentile_rank = index / len(sorted_arr)
print('percentile to which 10800 corresponds: ', percentile_rank)

fig, ax = plt.subplots()
# title of the plot
ax.set_title('Monte Carlo Simulation of a candidate project')
# Plot the histogram of the monte carlo simulation of the fourth project
ax.hist(mcs_results2[0][3], bins=200, color='grey', label='Histogram')
# title of the x axis
ax.set_xlabel('Cost in k€')
# Create a twin Axes object that shares the x-axis of the original Axes object
ax2 = ax.twinx()
# Plot the histogram of the monte carlo simulation of the first project in the form of a cumulative distribution function
ax2.hist(mcs_results2[0][3], bins=200, color='black', cumulative=True, histtype='step', density=True, label='Cumulative Distribution')
# Set the y-axis of the twin Axes object to be visible
ax2.yaxis.set_visible(True)
#set maximum value of the y axis of the twin Axes object to 1
ax2.set_ylim(0, 1)
# add grid to the plot following the y axis of the twin Axes object
ax2.grid(axis='y')
# add grid to the plot following the x axis of the original Axes object
ax.grid(axis='x')
# Add legend
ax.legend(loc='center left')
ax2.legend(loc='upper left')
plt.show()
#print(mcs_results2[0][3])
#print(len(mcs_results2[0][3]))
