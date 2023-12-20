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
import sys

#from copulas.multivariate import GaussianMultivariate
#from scipy.stats import rv_continuous, rv_histogram, norm, uniform, multivariate_normal, beta
#from fitter import Fitter, get_common_distributions, get_distributions

#import created scripts:
from task_rnd_triang_with_interrupts_stdev_new_R2 import *
from functions_for_simheuristic_v12 import *

# Redirect optput to a file instead of Terminal
# Save current standard output
stdout = sys.stdout

# Redirect standard output to a file
sys.stdout = open('output.txt', 'w')

# create an empty list to store the timestamps and labels
timestamps = []

start_time = time.time()
timestamps.append(('t = 0', time.time()))

#get budgetting confidence policy
#budgetting_confidence_policies = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
budgetting_confidence_policies = [0.75]
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
npv_results = []
budgets = []

#defining the correlation matrix to be used in the monte carlo simulation (and as check when the correlations are expected to be 0)
correlation_matrix = []


#*****


#I define the number of candidates to be considered and the number of iterations for the MCS
nrcandidates = 20
iterations = 1000
iterations_finalMCS = 10000

# iterations = 20
# iterations_finalMCS = 50

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

# mcs_results1[0] corresponds to the project costs and mcs_results1[1] to the project benefits (NPV)
x_perproj_matrix1 = pointestimate(mcs_results1[0], mcs_results1[1], budgetting_confidence_policies, nrcandidates)
print ("x_perproj_matrix1: ", x_perproj_matrix1)

# write the first timestamp and label to the list
timestamps.append(('First MCS with point estimate of budgets and NPV for each project', time.time()))

# extract first column of the matrix to get the budgeted costs of each project and store it in bdgtperproject_matrix
bdgtperproject_matrix = x_perproj_matrix1[0]
# extract second column of the matrix to get the NPV of each project and store it in npvperproject_matrix
npvperproject_matrix = x_perproj_matrix1[1]
# print("bdgtperproject_matrix at MAIN: ", bdgtperproject_matrix)
# print("npvperproject_matrix at MAIN: ", npvperproject_matrix)
# print("x_perproj_matrix1: ", x_perproj_matrix1)
# sum the costs of all projects to get the total cost of the portfolio if choosing all projects
totalcost = np.sum(x_perproj_matrix1[0])


# print("total portfolio cost allocation request (without correlations because it is a request):")
# print(totalcost)

# Defining the fitness function
def evaluate(individual, bdgtperproject, npvperproject, maxbdgt):
    total_cost = 0
    total_npv = 0
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
    for i in range(pf_cost10r.__len__()):
        if pf_cost10r[i] > maxbdgt:
            count = count + 1
    #array storing the portfolio risk not to exceed 3.800 Mio.€, as per-one risk units
    portfolio_confidence = 1-count/iterations
    # print("portfolio confidence:")
    # print(portfolio_confidence)
    for i in range(nrcandidates):
        if individual[i] == 1:
            total_cost += bdgtperproject[i]
            #total_cost += PROJECTS[i][0]
            # add the net present value of the project to the total net present value of the portfolio
            total_npv += npvperproject[i]
            #total_npv += npv[i][1]
    if total_cost > maxbdgt or portfolio_confidence < min_pf_conf:
        return 0, 0
    return total_npv, portfolio_confidence

# defining the function that maximizes the net present value of a portfolio of projects, while respecting the budget constraint (using a genetic algorithm)
def maximize_npv():
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
        print(f"Generation {generation}: Max NPV = {record['max']}")

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


# # correlation matrix (rnd) to be used in the next mcs simulation
# seed_value = 1005    
# np.random.seed(seed_value)  
# # Generate a random symmetric matrix  
# A = np.random.rand(initcandidates, initcandidates)  
# A = (A + A.T) / 2  
# # Compute the eigenvalues and eigenvectors of the matrix  
# eigenvalues, eigenvectors = np.linalg.eigh(A)  
# # Ensure the eigenvalues are positive  
# eigenvalues = np.abs(eigenvalues)  
# # Normalize the eigenvalues so that their sum is equal to nrcandidates  
# eigenvalues = eigenvalues / eigenvalues.sum() * initcandidates  
# # Compute the correlation matrix. Forcing positive values, as long as negative correlations are not usual in reality of projects  
# cm10r = np.abs(eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T))  
# # Ensure the diagonals are equal to 1  
# for i in range(initcandidates):  
#     cm10r[i, i] = 1

# df10r = correlatedMCS(mcs_results1, iterations, nrcandidates, initial_projection_indexes)
# print("df10r: ", df10r)

# I will test the correlated MCS 2 with different correlation values. I will initialize the experiment array with values from 0.1 till 0.9 (0.2spc)
correlations_for_experiment = np.arange(0, 1, 0.1)
npv_results = []
budgets = []
pf_cost20r = []
portfolios_exp = []
npv_results_exp = []
budgets_exp = []
confidences_exp = []
for z in range(len(correlations_for_experiment)):
    print("correlation matrix used in the experiment: ")
    print(correlations_for_experiment[z])
    cm10r = np.full((initcandidates, initcandidates), correlations_for_experiment[z])
    np.fill_diagonal(cm10r, 1)
    print(cm10r)
    #print(df10r)

    df10r = correlatedMCS2(mcs_results1, iterations, nrcandidates, initial_projection_indexes, cm10r)
    # write the second timestamp (substract the current time minus the previously stored timestamp) and label to the list
    timestamps.append(('First MCS with correlated cost and NPV for each project', time.time()))

    # Define the genetic algorithm parameters
    POPULATION_SIZE = 300 #was 180 #was 100 #was 50
    # POPULATION_SIZE = 18
    P_CROSSOVER = 0.4
    P_MUTATION = 0.6
    MAX_GENERATIONS = 1000 #was 500 #was 200 #was 100
    # MAX_GENERATIONS = 50
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
    toolbox.register("evaluate", evaluate, bdgtperproject=bdgtperproject_matrix, npvperproject=npvperproject_matrix, maxbdgt=maxbdgt)
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


    # this function calculates the npv of each project and then uses the maximizer function to obtain and return portfolio, npv and bdgt in a matrix (solutions)
    for ii in range(len(budgetting_confidence_policies)):
        # I take the column of bdgtperproject_matrix that corresponds to the budgetting confidence policy
        bdgtperproject=bdgtperproject_matrix[:,ii]
        #bdgtperproject=bdgtperproject_matrix[ii]
        # print(bdgtperproject)
        npvperproject=npvperproject_matrix[:,ii]
        #npvperproject=npvperproject_matrix[ii]
        # print(npvperproject)
        # execute the maximizer function to obtain the portfolio, and its npv and bdgt
        projectselection = maximize_npv()
        # assign the result from projectselection to the variable solutions
        solutions.append(projectselection)
        #print(solutions)
    # lately I only had one BCP, si it has performed the append only once, however as the solution is a hall of fame, it has appended a list of 3 individuals

    #store the npv results, portfolio results, portfolio confidence levels and budgets taken in different lists
    npv_results = [0] * len(projectselection)
    portfolio_results = [0] * len(projectselection)
    portfolio_confidence_levels = [0] * len(projectselection)
    pf_conf2 = [0] * len(projectselection)
    budgets = [0] * len(projectselection)
    for i in range(nrcandidates):
        npv_results = [[x[i].fitness.values[0][0] for x in solutions] for i in range(len(projectselection))]
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
    print ("npv_results: ", npv_results)
    print ("portfolio_results: ", projectselection)
    print ("portfolio_confidence_levels: ", portfolio_confidence_levels)
    print ("budgets: ", budgets)
    print ("portfolio_projection: ", portfolio_projection)
    print ("Indexes of selected projects: ", zipped_projection_indexes)
    print ("Number of candidate projects for stage 2: ", projected_candidates)

    print ("************ STARTING STAGE 2 (long MCS) **********")
    #second simulation to get all cdfs for cost & benefits after optimization step (may_update: was 1000)
    mcs_results2 = simulate(portfolio_projection,iterations_finalMCS)


    # mcs_results2[0] corresponds to the project costs and mcs_results2[1] to the project benefits (NPV)
    x_perproj_matrix2 = pointestimate(mcs_results2[0], mcs_results2[1], budgetting_confidence_policies, projected_candidates)
    # print ("x_perproj_matrix2: ", x_perproj_matrix2)

    # write the fourth timestamp and label to the list
    timestamps.append(('Second MCS, also including point estimate of budgets and NPV for shortlisted projects', time.time()))

    # we assume correlations at the cost side, not at the benefits side (conservative approach)
    # update x_perproj_matrix2 with the correlation effect registered inside df20r
    # print("x_perproj_matrix2: ", x_perproj_matrix2)
    # separate the budget and npv results from the x_perproj_matrix
    bdgtperproject_matrix2 = x_perproj_matrix2[0]
    npvperproject_matrix2 = x_perproj_matrix2[1]
    # print(type(bdgtperproject_matrix))
    # print(type(npvperproject_matrix))
    bdgtperproject_matrix_sqz = np.squeeze(bdgtperproject_matrix2)
    npvperproject_matrix_sqz = np.squeeze(npvperproject_matrix2)

    # remove all data that has zeroes from bdgtperproject_matrix and npvperproject_matrix
    # bdgtperproject_matrix = bdgtperproject_matrix[np.nonzero(bdgtperproject_matrix.flatten())]
    # npvperproject_matrix = npvperproject_matrix[np.nonzero(npvperproject_matrix.flatten())]

    # print("bdgtperproject_matrix: ", bdgtperproject_matrix)
    # print("npvperproject_matrix: ", npvperproject_matrix)
    print("size of bdgtperproject_matrix", len(bdgtperproject_matrix_sqz))
    print("size of npvperproject_matrix", len(npvperproject_matrix_sqz))
    print("size of mcs_results2", len(mcs_results2))

    # print("mcs_results2 (input para correlacionar): ", mcs_results2)

    # for each of the options obtained in projectselection, calculate the total portfolio npv and the portfolio budget based on the information from x_perproj_matrix
    npv_results = [0] * len(projectselection) # as many as len(projectselection) because we have one npv per item in HoF
    budgets = [0] * len(projectselection)
    pf_conf2 = [0] * len(projectselection)
    widened_bdgtperproject_matrix = [0] * nrcandidates # as many as initial amount of project candidates
    widened_npvperproject_matrix = [0] * nrcandidates
    # initialize dataframe called widened_df20r as a copy of df10r
    widened_df20r = df10r.copy()
    # enlarge the dataframe to the size of iterations_finalMCS
    widened_df20r = widened_df20r.reindex(range(iterations_finalMCS))
    # fill the dataframe with zeroes
    widened_df20r.iloc[:, :] = 0

    # print("correlation matrix used in the experiment: ")
    # print(correlations_for_experiment[z])
    # cm20r = np.full((len(bdgtperproject_matrix), len(bdgtperproject_matrix)), correlations_for_experiment[z])
    # np.fill_diagonal(cm20r, 1)
    # print(cm20r)
    # #print(df10r)

    df20r = correlatedMCS2(mcs_results2, iterations_finalMCS, projected_candidates, zipped_projection_indexes, cm10r)
    # print("df20r: ", df20r)

    # pick in order the values from bdgtperproject_matrix and npvperproject_matrix and store them in widened_bdgtperproject_matrix and widened_npvperproject_matrix
    # The location of the values to be picked is available in zipped_projection_indexes
    j=0
    for i in range(nrcandidates):
        if i in zipped_projection_indexes:
            widened_bdgtperproject_matrix [i] = round(bdgtperproject_matrix_sqz [j],3)
            widened_npvperproject_matrix [i] = round(npvperproject_matrix_sqz [j],3)
            j+=1
        else:
            pass
    # print("widened_bdgtperproject_matrix: ", widened_bdgtperproject_matrix)
    # print("widened_npvperproject_matrix: ", widened_npvperproject_matrix)

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
        #calculate the total portfolio npv by multiplying the npv of each project by the binary array obtained in projectselection
        npv_results[i] = np.sum(np.multiply(widened_npvperproject_matrix,projectselection[i]))
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
    finalsol_df = pd.DataFrame({'Portfolio': projectselection, 'Portfolio NPV': npv_results, 'Portfolio Budget': budgets, 'Portfolio confidence': pf_conf2})
    # order the dataframe by the portfolio npv, starting with the highest npv
    finalsol_df = finalsol_df.sort_values(by=['Portfolio NPV'], ascending=False)
    print ("Final Solution: ", finalsol_df)

    # write the fifth timestamp and label to the list. Calculation FINALIZED
    timestamps.append(('Application of correlation effect to final options', time.time()))

    segments = [0] * (len(timestamps)-1)

    # calculate the difference between each pair of timestamps
    for i in range(0, len(timestamps)-1):
        segments[i] = (timestamps[i+1][0], round(timestamps[i+1][1] - timestamps[i][1], 2))
        print(segments)
        
    # create a dataframe from the list of timestamps
    # crono_frame = pd.DataFrame(segments, columns=['Checkpoint', 'Execution time (s)'])

    # add a final register with the total execution time
    # crono_frame.loc['Total'] = ['Total', crono_frame['Execution time (s)'].sum()]

    # print the dataframe
    # print(crono_frame)

    # npv_results = []
    # budgets = []
    # pf_cost20r = []
    #pf_conf2 = []

    #from the sorted dataframe, take the first row, which corresponds to the highest npv portfolio and extract the data needed for the following pictures
    finalsol_df = finalsol_df.iloc[0]
    portfolio_results = finalsol_df[0]
    portfolios_exp.insert(z, portfolio_results)
    npv_results_escalar = finalsol_df[1]
    # the value corresponding to position z is going to be npv_results[z] and copy the value from npv_results_escalar to npv_results[z]
    npv_results_exp.insert(z, npv_results_escalar)
    #npv_results.append(finalsol_df[1])
    budgets_escalar = finalsol_df[2]
    budgets_exp.insert(z, budgets_escalar)
    confidences_escalar = finalsol_df[3]
    confidences_exp.insert(z, confidences_escalar)
    #budgets.append(finalsol_df[2])
    print("portfolio_results: ", portfolios_exp)
    print("npv_results: ", npv_results_exp)
    print("budgets: ", budgets_exp)

    #*** Total execution time
    print("Total execution time: %s seconds" %((time.time() - start_time)))

#separate the npv results from the solutions list
#npv_results = [round(x[1][0], 0) for x in solutions]
#separate the portfolio results from the solutions list
#portfolio_results = [x[0] for x in solutions]
#separate the budgets taken from the solutions list (was budgets = [x[2][0] for x in solutions] -> [0] PARA CUANDO SEA SOLO UN BCP
#budgets = [x[2][0] for x in solutions]

#DESACTIVAR ALL THIS SI QUIERES MIRAR TODOS JUNTOS - HASTA PLT(SHOW)
#represent in a scatter plot the results of optimal npv extracted from dataframe finalsol_df vs budgetting confidence policy
plt.figure(1)
plt.scatter(correlations_for_experiment, npv_results_exp, color='blue')
#zoom in the plot so that the minumum value of the x axis is 0.5 and the maximum value of the x axis is 1
plt.title("PV vs Correlation level of candidate portfolio")
plt.xlabel("Correlation Level")
plt.ylabel("PV (k€)", color='blue')
#add a second axis in green color with the confidence levels
ax1 = plt.gca()
ax1.set_facecolor('white')  # Set the face color to white
plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
ax2 = plt.gca().twinx()
ax1.set_facecolor('white')  # Set the face color to white
ax2.scatter(correlations_for_experiment, confidences_exp, color='green')
# add label "Reliability level" to the second axis
ax2.set_ylabel('Reliability level', color='green')

# rescale all fonts to 16
plt.rcParams.update({'font.size': 14})
#add the values of the npv and reliability results to the plot as annotations and displaced vertically a 1% of the y axis
for i, txt in enumerate(npv_results_exp):
    txt = "{:,}".format(round(txt))
    ax1.annotate(txt, (correlations_for_experiment[i], npv_results_exp[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    ax2.annotate(round(confidences_exp[i], 2), (correlations_for_experiment[i], confidences_exp[i]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)

# For the y-axis of ax1
for label in ax1.yaxis.get_ticklabels():
    label.set_color('blue')
# For the y-axis of ax2
for label in ax2.yaxis.get_ticklabels():
    label.set_color('green')

# for i, txt in enumerate(npv_results_exp):
#     txt = "{:,}".format(round(txt))
#     plt.annotate(txt, (correlations_for_experiment[i], npv_results_exp[i]), textcoords="offset points", xytext=(0, 10), ha='center')
# plt.xlim(0, 1)
# add grid to the plot. 
plt.xticks(np.arange(0, 1.1, 0.1))  # Grid lines from 0 to 1.1 with a step of 0.1
# Force that the vertical lines of the grid are placed every 0.1 units


plt.figure(2)
plt.scatter(correlations_for_experiment, budgets_exp, color='blue')
#zoom in the plot so that the minumum value of the x axis is 0.5 and the maximum value of the x axis is 1
plt.title("Cost vs Correlation level of candidate portfolio")
plt.xlabel("Correlation Level")
plt.ylabel("Cost (k€)", color='blue')
# plt.gcf().set_facecolor('white')
#add a second axis with the confidence levels
ax1 = plt.gca()
ax1.set_facecolor('white')  # Set the face color to white
plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
ax2 = plt.gca().twinx()
ax2.set_facecolor('white')  # Set the face color to white
ax2.scatter(correlations_for_experiment, confidences_exp, color='green')
# add label "Reliability level" to the second axis
ax2.set_ylabel('Reliability level', color='green')



# rescale all fonts to 16
plt.rcParams.update({'font.size': 14})
#add the values of the npv results to the plot as annotations and displaced vertically a 1% of the y axis
for i, txt in enumerate(budgets_exp):
    txt = "{:,}".format(round(txt))
    ax1.annotate(txt, (correlations_for_experiment[i], budgets_exp[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    ax2.annotate(round(confidences_exp[i], 2), (correlations_for_experiment[i], confidences_exp[i]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.xlim(0, 1.1)
plt.ylim(0, 1.1)
# add grid to the plot
plt.xticks(np.arange(0, 1.1, 0.1))  # Grid lines from 0 to 1.1 with a step of 0.1
# plt.grid(color='black', linestyle='--', linewidth=0.5, alpha=0.5)
for label in ax1.yaxis.get_ticklabels():
    label.set_color('blue')
# For the y-axis of ax2
for label in ax2.yaxis.get_ticklabels():
    label.set_color('green')
plt.show()