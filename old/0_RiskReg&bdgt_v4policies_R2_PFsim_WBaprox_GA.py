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
POPULATION_SIZE = 100
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 500
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
toolbox.register("evaluate", evaluate)
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
    # return the optimal portfolio from the hall of fame their fitness and the total budget
    print(hall_of_fame[0], hall_of_fame[0].fitness.values[0], portfolio_totalbudget(hall_of_fame[0]))
    return hall_of_fame[0], hall_of_fame[0].fitness.values[0], portfolio_totalbudget(hall_of_fame[0])

#defining the function that, for each budgetting confidence policy, computes the budgeted duration
#of each project and the standard deviation of the budgeted duration (and the related budgeted cost)
#initialize an array of budgeted durations that is nrcandidates x len(budgetting_confidence_policies)
budgetedcosts = np.zeros((nrcandidates, len(budgetting_confidence_policies)))
#initialize an array of standard deviations that is sized as far as nrcandidates
stdevs = np.zeros((nrcandidates, 1))
for i in range(nrcandidates):
    iterations=10000
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

#*** execution time
print("Execution time after SIMULATION step: %s milli-seconds" %((time.time() - start_time)* 1000))

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

#*** execution time
print("Execution time after OPTIMIZATION step: %s milli-seconds" %((time.time() - start_time)* 1000))

#DESACTIVAR ALL THIS SI QUIERES MIRAR TODOS JUNTOS - HASTA PLT(SHOW)
plt.figure(1)
plt.scatter(budgetting_confidence_policies, npv_results, color='grey')
#zoom in the plot so that the minumum value of the x axis is 0.5 and the maximum value of the x axis is 1
plt.title("NPV vs Budgetting Confidence Policy")
plt.xlabel("Budgetting Confidence Policy")
plt.ylabel("NPV")
# rescale all fonts to 16
plt.rcParams.update({'font.size': 14})
#add the values of the npv results to the plot as annotations and displaced vertically a 1% of the y axis
for i, txt in enumerate(npv_results):
    txt = "{:,}".format(round(txt))
    plt.annotate(txt, (budgetting_confidence_policies[i], npv_results[i]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.xlim(0.45, 1)
plt.grid()		
#plt.show()

# create a square array with the information included in portfolio_results
solution_portfolios = np.array(portfolio_results)
# plot the square array as a heatmap
#plt.figure(2)
fig, ax = plt.subplots()
plt.imshow(solution_portfolios, cmap='binary', interpolation='nearest', vmin=0, vmax=1)
plt.xlabel("Project", fontsize=12)
plt.ylabel("Budgetting Confidence Policy", fontsize=14)
plt.yticks(range(len(budgetting_confidence_policies)), budgetting_confidence_policies, fontsize=12)
plt.xticks(np.arange(0, nrcandidates, 1), fontsize=12)
plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

for i, budget in enumerate(budgets):
    plt.text(nrcandidates + 0.5, i, "${:.2f}".format(budget), ha='left', va='center', fontsize=14)

plt.text(nrcandidates + 2, len(budgetting_confidence_policies) / 2, "Portfolio Budget", ha='center', va='center', rotation=270, fontsize=14)
plt.tight_layout()


#calculate the variance of the portfolio for each budgetting confidence policy by taking the sum of the squared standard deviations of each project
portfolio_var = sum(stdevs**2)
#calculate the standard deviation of the portfolio for each budgetting confidence policy by taking the square root of the variance
portfolio_stdev = np.sqrt(portfolio_var)
print(solution_portfolios)
print(portfolio_var)
print(portfolio_stdev)

#extract the sixth portfolio included in array portfolio_results
chosen_portfolio = portfolio_results[6]
#multiply dataframe 0 by the chosen portfolio to reflect the effect of the projects that are chosen
pf_df = df0 * chosen_portfolio
#sum the rows of the new dataframe to calculate the total cost of the portfolio
pf_cost = pf_df.sum(axis=1)

fig, ax = plt.subplots()
# title of the plot
# ax.set_title('Monte Carlo Simulation of a candidate project')
# Plot the histogram of the monte carlo simulation of the first project
ax.hist(mcs_results[0], bins=200, color='grey', label='Histogram')
# title of the x axis
ax.set_xlabel('Cost in k€')
# Create a twin Axes object that shares the x-axis of the original Axes object
ax2 = ax.twinx()
# Plot the histogram of the monte carlo simulation of the first project in the form of a cumulative distribution function
ax2.hist(mcs_results[0], bins=200, color='black', cumulative=True, histtype='step', density=True, label='Cumulative Distribution')
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



#plot the histogram of the resulting costs
plt.figure(4)
plt.hist(pf_cost, bins=200, color = 'grey' )
plt.title("Histogram of the resulting costs obtained directly from MCS")
#zoom x axis so that the histogram is more visible
plt.xlim(min(pf_cost)-10, max(pf_cost)+10)
#zoom y axis so that the histogram is more visible
#extract the maximum of the resulting costs
maxcost = max(pf_cost)
#count how many results were higher than maxbdgt
count = 0
for i in range(pf_cost.__len__()):
    if pf_cost[i] > maxbdgt:
        count = count + 1
portfolio_risk = np.zeros(5)
portfolio_risk[0] = (1-count/iterations)

# Correlation matrix to be used in the next mcs simulation
cm109 = np.full((10, 10), 0.9)
np.fill_diagonal(cm109, 1)

# Correlation matrix to be used in the next mcs simulation
cm106 = np.full((10, 10), 0.6)
np.fill_diagonal(cm106, 1)

# Correlation matrix to be used in the next mcs simulation
cm103 = np.full((10, 10), 0.3)
np.fill_diagonal(cm103, 1)


# *********Correlation matrix with random values between 0 and 1, but positive semidefinite***************
# Generate a random symmetric matrix
A = np.random.rand(10, 10)
A = (A + A.T) / 2
# Compute the eigenvalues and eigenvectors of the matrix
eigenvalues, eigenvectors = np.linalg.eigh(A)
# Ensure the eigenvalues are positive
eigenvalues = np.abs(eigenvalues)
# Normalize the eigenvalues so that their sum is equal to 10
eigenvalues = eigenvalues / eigenvalues.sum() * 10
# Compute the covariance matrix. Forcing positive values, as long as negative correlations are not usual in reality of projects
cm10r = np.abs(eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T))
# Ensure the diagonals are equal to 1
for i in range(10):
    cm10r[i, i] = 1
print('cm10r:')
print(cm10r)



#initialize dataframe df109 with size nrcandidates x iterations
df109 = pd.DataFrame(np.zeros((iterations, nrcandidates)))
# step 1: draw random variates from a multivariate normal distribution 
# with the targeted correlation structure
r0 = [0] * cm109.shape[0]                       # create vector r with as many zeros as correlation matrix has variables (row or columns)
mv_norm = multivariate_normal(mean=r0, cov=cm109)    # means = vector of zeros; cov = targeted corr matrix
rand_Nmv = mv_norm.rvs(iterations)                               # draw N random variates
# step 2: convert the r * N multivariate variates to scores 
rand_U = norm.cdf(rand_Nmv)   # use its cdf to generate N scores (probabilities between 0 and 1) from the multinormal random variates
# step 3: instantiate the 10 marginal distributions 
d_P1 = beta(a[0], b[0], loc[0], scale[0])
d_P2 = beta(a[1], b[1], loc[1], scale[1])
d_P3 = beta(a[2], b[2], loc[2], scale[2])
d_P4 = beta(a[3], b[3], loc[3], scale[3])
d_P5 = beta(a[4], b[4], loc[4], scale[4])
d_P6 = beta(a[5], b[5], loc[5], scale[5])
d_P7 = beta(a[6], b[6], loc[6], scale[6])
d_P8 = beta(a[7], b[7], loc[7], scale[7])
d_P9 = beta(a[8], b[8], loc[8], scale[8])
d_P10 = beta(a[9], b[9], loc[9], scale[9])
# step 4: apply the inverse of the marginal cdfs to the scores
# draw N random variates for each of the three marginal distributions
# WITHOUT applying a copula
rand_P1 = d_P1.rvs(iterations)
rand_P2 = d_P2.rvs(iterations)
rand_P3 = d_P3.rvs(iterations)
rand_P4 = d_P4.rvs(iterations)
rand_P5 = d_P5.rvs(iterations)
rand_P6 = d_P6.rvs(iterations)
rand_P7 = d_P7.rvs(iterations)
rand_P8 = d_P8.rvs(iterations)
rand_P9 = d_P9.rvs(iterations)
rand_P10 = d_P10.rvs(iterations)
# initial correlation structure before applying a copula
c_before = np.corrcoef([rand_P1, rand_P2, rand_P3, rand_P4, rand_P5, rand_P6, rand_P7, rand_P8, rand_P9, rand_P10])
# step 4: draw N random variates for each of the three marginal didsibutions
# and use as inputs the correlated uniform scores we have generated in step 2
rand_P1 = d_P1.ppf(rand_U[:, 0])
rand_P2 = d_P2.ppf(rand_U[:, 1])
rand_P3 = d_P3.ppf(rand_U[:, 2])
rand_P4 = d_P4.ppf(rand_U[:, 3])
rand_P5 = d_P5.ppf(rand_U[:, 4])
rand_P6 = d_P6.ppf(rand_U[:, 5])
rand_P7 = d_P7.ppf(rand_U[:, 6])
rand_P8 = d_P8.ppf(rand_U[:, 7])
rand_P9 = d_P9.ppf(rand_U[:, 8])
rand_P10 = d_P10.ppf(rand_U[:, 9])
# final correlation structure after applying a copula
c_after = np.corrcoef([rand_P1, rand_P2, rand_P3, rand_P4, rand_P5, rand_P6, rand_P7, rand_P8, rand_P9, rand_P10])
print("Correlation matrix before applying a copula:")
print(c_before)
print("Correlation matrix after applying a copula:")
print(c_after)
# step 5: store the N random variates in the dataframe
df109[0] = rand_P1
df109[1] = rand_P2
df109[2] = rand_P3
df109[3] = rand_P4
df109[4] = rand_P5
df109[5] = rand_P6
df109[6] = rand_P7
df109[7] = rand_P8
df109[8] = rand_P9
df109[9] = rand_P10
df109.rename(columns={0:"P01", 1:"P02", 2:"P03", 3:"P04", 4:"P05", 5:"P06", 6:"P07", 7:"P08", 8:"P09", 9:"P10"}, inplace=True)
#print('df1_09')
#print(df1_09)
print('df109')
print(df109)
#multiply dataframe 109 by the chosen portfolio to reflect the effect of the projects that are chosen
pf_df109 = df109 * chosen_portfolio
#sum the rows of the new dataframe to calculate the total cost of the portfolio
pf_cost109 = pf_df109.sum(axis=1)
#plot the histogram of the resulting costs
#plt.figure(4)
# Create a 2x2 subplot grid
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
ax[0, 0].hist(pf_cost, bins=200, color = 'grey', range=(3000, 4800))
ax[0, 0].hist(pf_cost109, bins=200, color = 'black', histtype="step")

#extract the maximum of the resulting costs
maxcost109 = max(pf_cost109)
print("max cost:")
print(maxcost109)
#count how many results were higher than maxbdgt
count = 0
for i in range(pf_cost109.__len__()):
    if pf_cost109[i] > maxbdgt:
        count = count + 1
#array storing the portfolio risk not to exceed 3.800 Mio.€, as per-one risk units
portfolio_risk[1] = 1-count/iterations


#initialize dataframe df106 with size nrcandidates x iterations
df106 = pd.DataFrame(np.zeros((iterations, nrcandidates)))
# step 1: draw random variates from a multivariate normal distribution 
# with the targeted correlation structure
r0 = [0] * cm106.shape[0]                       # create vector r with as many zeros as correlation matrix has variables (row or columns)
mv_norm = multivariate_normal(mean=r0, cov=cm106)    # means = vector of zeros; cov = targeted corr matrix
rand_Nmv = mv_norm.rvs(iterations)                               # draw N random variates
# step 2: convert the r * N multivariate variates to scores 
rand_U = norm.cdf(rand_Nmv)   # use its cdf to generate N scores (probabilities between 0 and 1) from the multinormal random variates
# step 3: instantiate the 10 marginal distributions 
d_P1 = beta(a[0], b[0], loc[0], scale[0])
d_P2 = beta(a[1], b[1], loc[1], scale[1])
d_P3 = beta(a[2], b[2], loc[2], scale[2])
d_P4 = beta(a[3], b[3], loc[3], scale[3])
d_P5 = beta(a[4], b[4], loc[4], scale[4])
d_P6 = beta(a[5], b[5], loc[5], scale[5])
d_P7 = beta(a[6], b[6], loc[6], scale[6])
d_P8 = beta(a[7], b[7], loc[7], scale[7])
d_P9 = beta(a[8], b[8], loc[8], scale[8])
d_P10 = beta(a[9], b[9], loc[9], scale[9])
# draw N random variates for each of the three marginal distributions
# WITHOUT applying a copula
rand_P1 = d_P1.rvs(iterations)
rand_P2 = d_P2.rvs(iterations)
rand_P3 = d_P3.rvs(iterations)
rand_P4 = d_P4.rvs(iterations)
rand_P5 = d_P5.rvs(iterations)
rand_P6 = d_P6.rvs(iterations)
rand_P7 = d_P7.rvs(iterations)
rand_P8 = d_P8.rvs(iterations)
rand_P9 = d_P9.rvs(iterations)
rand_P10 = d_P10.rvs(iterations)
# initial correlation structure before applying a copula
c_before = np.corrcoef([rand_P1, rand_P2, rand_P3, rand_P4, rand_P5, rand_P6, rand_P7, rand_P8, rand_P9, rand_P10])
# step 4: draw N random variates for each of the three marginal didsibutions
# and use as inputs the correlated uniform scores we have generated in step 2
rand_P1 = d_P1.ppf(rand_U[:, 0])
rand_P2 = d_P2.ppf(rand_U[:, 1])
rand_P3 = d_P3.ppf(rand_U[:, 2])
rand_P4 = d_P4.ppf(rand_U[:, 3])
rand_P5 = d_P5.ppf(rand_U[:, 4])
rand_P6 = d_P6.ppf(rand_U[:, 5])
rand_P7 = d_P7.ppf(rand_U[:, 6])
rand_P8 = d_P8.ppf(rand_U[:, 7])
rand_P9 = d_P9.ppf(rand_U[:, 8])
rand_P10 = d_P10.ppf(rand_U[:, 9])
# final correlation structure after applying a copula
c_after = np.corrcoef([rand_P1, rand_P2, rand_P3, rand_P4, rand_P5, rand_P6, rand_P7, rand_P8, rand_P9, rand_P10])
print("Correlation matrix before applying a copula:")
print(c_before)
print("Correlation matrix after applying a copula:")
print(c_after)
# step 5: store the N random variates in the dataframe
df106[0] = rand_P1
df106[1] = rand_P2
df106[2] = rand_P3
df106[3] = rand_P4
df106[4] = rand_P5
df106[5] = rand_P6
df106[6] = rand_P7
df106[7] = rand_P8
df106[8] = rand_P9
df106[9] = rand_P10
df106.rename(columns={0:"P01", 1:"P02", 2:"P03", 3:"P04", 4:"P05", 5:"P06", 6:"P07", 7:"P08", 8:"P09", 9:"P10"}, inplace=True)

#multiply dataframe 106 by the chosen portfolio to reflect the effect of the projects that are chosen
pf_df106 = df106 * chosen_portfolio
#sum the rows of the new dataframe to calculate the total cost of the portfolio
pf_cost106 = pf_df106.sum(axis=1)
#plot the histogram of the resulting costs as another frame inside figure 4
#plt.figure(4)
ax[0, 1].hist(pf_cost, bins=200, color = 'grey', range=(3000, 4800))
ax[0, 1].hist(pf_cost106, bins=200, color = 'black', histtype="step")

#extract the maximum of the resulting costs
maxcost106 = max(pf_cost106)
print("max cost:")
print(maxcost106)
#count how many results were higher than maxbdgt
count = 0
for i in range(pf_cost106.__len__()):
    if pf_cost106[i] > maxbdgt:
        count = count + 1
#array storing the portfolio risk not to exceed 3.800 Mio.€, as per-one risk units
portfolio_risk[2] = 1-count/iterations

#initialize dataframe df103 with size nrcandidates x iterations
df103 = pd.DataFrame(np.zeros((iterations, nrcandidates)))
# step 1: draw random variates from a multivariate normal distribution 
# with the targeted correlation structure
r0 = [0] * cm103.shape[0]                       # create vector r with as many zeros as correlation matrix has variables (row or columns)
mv_norm = multivariate_normal(mean=r0, cov=cm103)    # means = vector of zeros; cov = targeted corr matrix
rand_Nmv = mv_norm.rvs(iterations)                               # draw N random variates
# step 2: convert the r * N multivariate variates to scores 
rand_U = norm.cdf(rand_Nmv)   # use its cdf to generate N scores (probabilities between 0 and 1) from the multinormal random variates
# step 3: instantiate the 10 marginal distributions 
d_P1 = beta(a[0], b[0], loc[0], scale[0])
d_P2 = beta(a[1], b[1], loc[1], scale[1])
d_P3 = beta(a[2], b[2], loc[2], scale[2])
d_P4 = beta(a[3], b[3], loc[3], scale[3])
d_P5 = beta(a[4], b[4], loc[4], scale[4])
d_P6 = beta(a[5], b[5], loc[5], scale[5])
d_P7 = beta(a[6], b[6], loc[6], scale[6])
d_P8 = beta(a[7], b[7], loc[7], scale[7])
d_P9 = beta(a[8], b[8], loc[8], scale[8])
d_P10 = beta(a[9], b[9], loc[9], scale[9])
# draw N random variates for each of the three marginal distributions
# WITHOUT applying a copula
rand_P1 = d_P1.rvs(iterations)
rand_P2 = d_P2.rvs(iterations)
rand_P3 = d_P3.rvs(iterations)
rand_P4 = d_P4.rvs(iterations)
rand_P5 = d_P5.rvs(iterations)
rand_P6 = d_P6.rvs(iterations)
rand_P7 = d_P7.rvs(iterations)
rand_P8 = d_P8.rvs(iterations)
rand_P9 = d_P9.rvs(iterations)
rand_P10 = d_P10.rvs(iterations)
# initial correlation structure before applying a copula
c_before = np.corrcoef([rand_P1, rand_P2, rand_P3, rand_P4, rand_P5, rand_P6, rand_P7, rand_P8, rand_P9, rand_P10])
# step 4: draw N random variates for each of the three marginal didsibutions
# and use as inputs the correlated uniform scores we have generated in step 2
rand_P1 = d_P1.ppf(rand_U[:, 0])
rand_P2 = d_P2.ppf(rand_U[:, 1])
rand_P3 = d_P3.ppf(rand_U[:, 2])
rand_P4 = d_P4.ppf(rand_U[:, 3])
rand_P5 = d_P5.ppf(rand_U[:, 4])
rand_P6 = d_P6.ppf(rand_U[:, 5])
rand_P7 = d_P7.ppf(rand_U[:, 6])
rand_P8 = d_P8.ppf(rand_U[:, 7])
rand_P9 = d_P9.ppf(rand_U[:, 8])
rand_P10 = d_P10.ppf(rand_U[:, 9])
# final correlation structure after applying a copula
c_after = np.corrcoef([rand_P1, rand_P2, rand_P3, rand_P4, rand_P5, rand_P6, rand_P7, rand_P8, rand_P9, rand_P10])
print("Correlation matrix before applying a copula:")
print(c_before)
print("Correlation matrix after applying a copula:")
print(c_after)
# step 5: store the N random variates in the dataframe
df103[0] = rand_P1
df103[1] = rand_P2
df103[2] = rand_P3
df103[3] = rand_P4
df103[4] = rand_P5
df103[5] = rand_P6
df103[6] = rand_P7
df103[7] = rand_P8
df103[8] = rand_P9
df103[9] = rand_P10
df103.rename(columns={0:"P01", 1:"P02", 2:"P03", 3:"P04", 4:"P05", 5:"P06", 6:"P07", 7:"P08", 8:"P09", 9:"P10"}, inplace=True)
#print('df1_09')
#print(df1_09)
print('df103')
print(df103)
#multiply dataframe 103 by the chosen portfolio to reflect the effect of the projects that are chosen
pf_df103 = df103 * chosen_portfolio
#sum the rows of the new dataframe to calculate the total cost of the portfolio
pf_cost103 = pf_df103.sum(axis=1)
#plot the histogram of the resulting costs
#plt.figure(4)
ax[1, 0].hist(pf_cost, bins=200, color = 'grey', range=(3000, 4800), label = 'uncorrelated histogram')
ax[1, 0].hist(pf_cost103, bins=200, color = 'black', histtype="step", label = 'correlated histogram')
# Set the common legend
fig.legend(loc='lower center', ncol=4)


#extract the maximum of the resulting costs
maxcost103 = max(pf_cost103)
print("max cost:")
print(maxcost103)
#count how many results were higher than maxbdgt
count = 0
for i in range(pf_cost103.__len__()):
    if pf_cost103[i] > maxbdgt:
        count = count + 1
#array storing the portfolio risk not to exceed 3.800 Mio.€, as per-one risk units
portfolio_risk[3] = 1-count/iterations


#initialize dataframe df10r with size nrcandidates x iterations
df10r = pd.DataFrame(np.zeros((iterations, nrcandidates)))
# step 1: draw random variates from a multivariate normal distribution 
# with the targeted correlation structure
r0 = [0] * cm10r.shape[0]                       # create vector r with as many zeros as correlation matrix has variables (row or columns)
mv_norm = multivariate_normal(mean=r0, cov=cm10r)    # means = vector of zeros; cov = targeted corr matrix
rand_Nmv = mv_norm.rvs(iterations)                               # draw N random variates
# step 2: convert the r * N multivariate variates to scores 
rand_U = norm.cdf(rand_Nmv)   # use its cdf to generate N scores (probabilities between 0 and 1) from the multinormal random variates
# step 3: instantiate the 10 marginal distributions 
d_P1 = beta(a[0], b[0], loc[0], scale[0])
d_P2 = beta(a[1], b[1], loc[1], scale[1])
d_P3 = beta(a[2], b[2], loc[2], scale[2])
d_P4 = beta(a[3], b[3], loc[3], scale[3])
d_P5 = beta(a[4], b[4], loc[4], scale[4])
d_P6 = beta(a[5], b[5], loc[5], scale[5])
d_P7 = beta(a[6], b[6], loc[6], scale[6])
d_P8 = beta(a[7], b[7], loc[7], scale[7])
d_P9 = beta(a[8], b[8], loc[8], scale[8])
d_P10 = beta(a[9], b[9], loc[9], scale[9])
# draw N random variates for each of the three marginal distributions
# WITHOUT applying a copula
rand_P1 = d_P1.rvs(iterations)
rand_P2 = d_P2.rvs(iterations)
rand_P3 = d_P3.rvs(iterations)
rand_P4 = d_P4.rvs(iterations)
rand_P5 = d_P5.rvs(iterations)
rand_P6 = d_P6.rvs(iterations)
rand_P7 = d_P7.rvs(iterations)
rand_P8 = d_P8.rvs(iterations)
rand_P9 = d_P9.rvs(iterations)
rand_P10 = d_P10.rvs(iterations)
# initial correlation structure before applying a copula
c_before = np.corrcoef([rand_P1, rand_P2, rand_P3, rand_P4, rand_P5, rand_P6, rand_P7, rand_P8, rand_P9, rand_P10])
# step 4: draw N random variates for each of the three marginal didsibutions
# and use as inputs the correlated uniform scores we have generated in step 2
rand_P1 = d_P1.ppf(rand_U[:, 0])
rand_P2 = d_P2.ppf(rand_U[:, 1])
rand_P3 = d_P3.ppf(rand_U[:, 2])
rand_P4 = d_P4.ppf(rand_U[:, 3])
rand_P5 = d_P5.ppf(rand_U[:, 4])
rand_P6 = d_P6.ppf(rand_U[:, 5])
rand_P7 = d_P7.ppf(rand_U[:, 6])
rand_P8 = d_P8.ppf(rand_U[:, 7])
rand_P9 = d_P9.ppf(rand_U[:, 8])
rand_P10 = d_P10.ppf(rand_U[:, 9])
# final correlation structure after applying a copula
c_after = np.corrcoef([rand_P1, rand_P2, rand_P3, rand_P4, rand_P5, rand_P6, rand_P7, rand_P8, rand_P9, rand_P10])
print("Correlation matrix before applying a copula:")
print(c_before)
print("Correlation matrix after applying a copula:")
print(c_after)
# step 5: store the N random variates in the dataframe
df10r[0] = rand_P1
df10r[1] = rand_P2
df10r[2] = rand_P3
df10r[3] = rand_P4
df10r[4] = rand_P5
df10r[5] = rand_P6
df10r[6] = rand_P7
df10r[7] = rand_P8
df10r[8] = rand_P9
df10r[9] = rand_P10
df10r.rename(columns={0:"P01", 1:"P02", 2:"P03", 3:"P04", 4:"P05", 5:"P06", 6:"P07", 7:"P08", 8:"P09", 9:"P10"}, inplace=True)

#multiply dataframe 10r by the chosen portfolio to reflect the effect of the projects that are chosen
pf_df10r = df10r * chosen_portfolio
#sum the rows of the new dataframe to calculate the total cost of the portfolio
pf_cost10r = pf_df10r.sum(axis=1)
#plot the histogram of the resulting costs
#plt.figure(4)
ax[1,1].hist(pf_cost, bins=200, color = 'grey', range=(3000, 4800))
ax[1,1].hist(pf_cost10r, bins=200, color = 'black', histtype="step")
ax[0, 0].set_title('Correlations: 0.9')
ax[0, 1].set_title('Correlations: 0.6')
ax[1, 0].set_title('Correlations: 0.3')
ax[1, 1].set_title('Random Correlations')


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
portfolio_risk[4] = 1-count/iterations

#print(df0)
#print(correlation_matrix0)
# plot the scatter matrix
pd.plotting.scatter_matrix(df0, alpha=0.2, figsize=(6, 6), diagonal='kde', color='grey', density_kwds={'color': 'grey'})
#plot the scatter matrix of df0 with seaborn pairplot function with grey color and a diagonal with a kde plot
#sns.pairplot(df0, diag_kind="kde", palette="Greys")
# add title and axis labels
plt.suptitle('Correlation matrix of the MCS results where all projects are fully independent (in k€)')
plt.xlabel('Projects and cost in k€')
plt.ylabel('Projects and cost in k€')
#plt.show()
# plot the scatter matrix
pd.plotting.scatter_matrix(df109, alpha=0.2, figsize=(6, 6), diagonal='kde', color='grey', density_kwds={'color': 'grey'})
# add title and axis labels
plt.suptitle('Correlation matrix of the MCS results where all projects are correlated by 0.9')
plt.xlabel('Projects and cost in k€')
plt.ylabel('Projects and cost in k€')
#plt.show()
# plot the scatter matrix
pd.plotting.scatter_matrix(df106, alpha=0.2, figsize=(6, 6), diagonal='kde', color='grey', density_kwds={'color': 'grey'})
# add title and axis labels
plt.suptitle('Correlation matrix of the MCS results where all projects are correlated by 0.6')
plt.xlabel('Projects and cost in k€')
plt.ylabel('Projects and cost in k€')
#plt.show()
# plot the scatter matrix
pd.plotting.scatter_matrix(df103, alpha=0.2, figsize=(6, 6), diagonal='kde', color='grey', density_kwds={'color': 'grey'})
# add title and axis labels
plt.suptitle('Correlation matrix of the MCS results where all projects are correlated by 0.3')
plt.xlabel('Projects and cost in k€')
plt.ylabel('Projects and cost in k€')
#plt.show()
# plot the scatter matrix
pd.plotting.scatter_matrix(df10r, alpha=0.2, figsize=(6, 6), diagonal='kde', color='grey', density_kwds={'color': 'grey'})
# add title and axis labels
plt.suptitle('Correlation matrix of the MCS results where all projs are randomly correlated')
plt.xlabel('Projects and cost in k€')
plt.ylabel('Projects and cost in k€')
#plt.show()

#convert the array of portfolio risks into a dataframe with header each of the correlation levels used
df_portfolio_risk = pd.DataFrame(portfolio_risk)
#transpose the dataframe
df_portfolio_risk = df_portfolio_risk.transpose()
#rename the columns of the dataframe
df_portfolio_risk.rename(columns={0:"0", 1:"0.9", 2:"0.6", 3:"0.3", 4:"random"}, inplace=True)
#current_cols = df_portfolio_risk.columns
print(df_portfolio_risk)



# Plot the portfolio risks
df_portfolio_risk.plot(kind='bar', title='Portfolio risks')
# Format the bars so that they have different patterns in order to be more visible
colors = ['black', 'dimgrey', 'grey', 'darkgrey', 'lightgrey']
fig, ax = plt.subplots()
for i, d in enumerate(df_portfolio_risk.values[0]):
    ax.bar(i, d, edgecolor='black', color=colors[i])
# Add y grid to the plot every 0.05
plt.yticks(np.arange(0, 1.05, 0.1))
## Add x labels to the plot
plt.xticks(np.arange(5), df_portfolio_risk.columns)
# Add y values to the plot
for i, d in enumerate(df_portfolio_risk.values[0]):
    plt.text(i-0.2, d+0.01, str(round(d,2)))
plt.grid(axis='y')
plt.show()

#make sure no legend appears in the next plot
plt.figure(12)
plt.legend().set_visible(False)
#heatmap of the correlation matrix cm10r
sns.set(font_scale=1.15)
sns.heatmap(cm10r, annot=True, cmap="Greys")

plt.show()
#*** execution time
print("Execution time: %s milli-seconds" %((time.time() - start_time)* 1000))


  



