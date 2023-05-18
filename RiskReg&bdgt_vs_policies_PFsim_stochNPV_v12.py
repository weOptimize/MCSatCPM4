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
#from copulas.multivariate import GaussianMultivariate
#from scipy.stats import rv_continuous, rv_histogram, norm, uniform, multivariate_normal, beta
#from fitter import Fitter, get_common_distributions, get_distributions

#import created scripts:
from task_rnd_triang_with_interrupts_stdev_new_R2 import *
from functions_for_simheuristic_v12 import *

start_time = time.time()

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
nrcandidates = 10
iterations = 100
iterations_finalMCS = 2000

#I define the budget constraint (in k€) and the minimum confidence level for the portfolio
maxbdgt = 3800
min_pf_conf = 0.90

#initialize an array of budgeted durations that is nrcandidates x len(budgetting_confidence_policies)
budgetedcosts = np.zeros((nrcandidates, len(budgetting_confidence_policies)))

#I define a candidate array of size nr candidates with all ones
candidatearray = np.ones(nrcandidates)

#first simulation to get all cdfs for cost & benefits before optimization step (may_update: was 1000)
mcs_results1 = simulate(candidatearray,iterations)
#print("mcs results1: ", mcs_results1[0])

#perform the point estimate of the cost (at each confidence level) and benefits of each project
#and store the results in a matrix
#initialize an array of budgeted durations that is nrcandidates x len(budgetting_confidence_policies)
#print(mcs_results1[0])
#print(mcs_results1[1])
# mcs_results1[0] corresponds to the project costs and mcs_results1[1] to the project benefits (NPV)
x_perproj_matrix1 = pointestimate(mcs_results1[0], mcs_results1[1], budgetting_confidence_policies)
# print("x_perproj_matrix1: ", x_perproj_matrix1)
# sum the costs of all projects to get the total cost of the portfolio if choosing all projects
totalcost = np.sum(x_perproj_matrix1[0])


print("total portfolio cost allocation request (without correlations because it is a request):")
print(totalcost)

df10r = correlatedMCS(mcs_results1, iterations)


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
    #print("portfolio confidence:")
    #print(portfolio_confidence)
    for i in range(nrcandidates):
        #print(total_cost)
        if individual[i] == 1:
            total_cost += bdgtperproject[i]
            #total_cost += PROJECTS[i][0]
            # add the net present value of the project to the total net present value of the portfolio
            total_npv += npvperproject[i]
            #total_npv += npv[i][1]
    if total_cost > maxbdgt or portfolio_confidence < min_pf_conf:
        return 0, 0
    return total_npv, portfolio_confidence

# Define the genetic algorithm parameters
POPULATION_SIZE = 30 #was 100 #was 50
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 60 #was 500 #was 200 #was 100
HALL_OF_FAME_SIZE = 3

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
    print(hall_of_fame[0], hall_of_fame[0].fitness.values[0], hall_of_fame[0].fitness.values[1], portfolio_totalbudget(hall_of_fame[0], bdgtperproject_matrix))
    print(hall_of_fame[1], hall_of_fame[1].fitness.values[0], hall_of_fame[1].fitness.values[1], portfolio_totalbudget(hall_of_fame[1], bdgtperproject_matrix))
    print(hall_of_fame[2], hall_of_fame[2].fitness.values[0], hall_of_fame[2].fitness.values[1], portfolio_totalbudget(hall_of_fame[2], bdgtperproject_matrix))
    #return hall_of_fame[0], hall_of_fame[0].fitness.values[0][0], portfolio_totalbudget(hall_of_fame[0], bdgtperproject_matrix)
    return hall_of_fame

# this function calculates the npv of each project and then uses the maximizer function to obtain and return portfolio, npv and bdgt in a matrix (solutions)
for i in range(len(budgetting_confidence_policies)):
    # I take the column of bdgtperproject_matrix that corresponds to the budgetting confidence policy
    bdgtperproject=bdgtperproject_matrix[:,i]
    # print(bdgtperproject)
    npvperproject=npvperproject_matrix[:,i]
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
for i in range(len(projectselection)):
    npv_results = [[x[i].fitness.values[0][0] for x in solutions] for i in range(len(projectselection))]
    #portfolio_results = [[x[i] for x in solutions] for i in range(len(projectselection))]
    portfolio_confidence_levels = [[x[i].fitness.values[1] for x in solutions] for i in range(len(projectselection))]
    budgets = [[portfolio_totalbudget(x[i], bdgtperproject_matrix)[0] for x in solutions] for i in range(len(projectselection))]

# take all arrays inside portfolio_results and sum all of them
portfolio_projection = [sum(x) for x in zip(*projectselection)]

# convert portfolio_projection array into a binary array, where 1 means that the project is selected and 0 means that it is not
portfolio_projection = [1 if x > 0 else 0 for x in portfolio_projection]

# convert portfolio_projection in a full ones array
# portfolio_projection = [1] * len(portfolio_projection)

print ("************ SUMMARY STAGE 1 **********")
print ("npv_results: ", npv_results)
print ("portfolio_results: ", projectselection)
print ("portfolio_confidence_levels: ", portfolio_confidence_levels)
print ("budgets: ", budgets)
print ("portfolio_projection: ", portfolio_projection)

print ("************ STARTING STAGE 2 (long MCS) **********")
#second simulation to get all cdfs for cost & benefits after optimization step (may_update: was 1000)
mcs_results2 = simulate(portfolio_projection,iterations_finalMCS)


# mcs_results2[0] corresponds to the project costs and mcs_results2[1] to the project benefits (NPV)
x_perproj_matrix2 = pointestimate(mcs_results2[0], mcs_results2[1], budgetting_confidence_policies)


# we assume correlations at the cost side, not at the benefits side (conservative approach)
# update x_perproj_matrix2 with the correlation effect registered inside df20r


# print("x_perproj_matrix2: ", x_perproj_matrix2)
# separate the budget and npv results from the x_perproj_matrix
bdgtperproject_matrix = x_perproj_matrix2[0]
npvperproject_matrix = x_perproj_matrix2[1]
#print(type(bdgtperproject_matrix))
#print(type(npvperproject_matrix))
bdgtperproject_matrix = np.squeeze(bdgtperproject_matrix)
npvperproject_matrix = np.squeeze(npvperproject_matrix)
print(bdgtperproject_matrix)
print(npvperproject_matrix)

# for each of the options obtained in projectselection, calculate the total portfolio npv and the portfolio budget based on the information from x_perproj_matrix
npv_results = [0] * len(projectselection)
budgets = [0] * len(projectselection)
pf_conf2 = [0] * len(projectselection)
df20r = correlatedMCS(mcs_results2, iterations_finalMCS)
for i in range(len(projectselection)):
    #calculate the total portfolio budget by multiplying the budget of each project by the binary array obtained in projectselection    
    print(projectselection[i])
    budgets[i] = np.sum(np.multiply(bdgtperproject_matrix,projectselection[i]))
    #calculate the total portfolio npv by multiplying the npv of each project by the binary array obtained in projectselection
    npv_results[i] = np.sum(np.multiply(npvperproject_matrix,projectselection[i]))
    #multiply dataframe 20r by the chosen portfolio to reflect the effect of the projects that are chosen
    pf_df20r = df20r * projectselection[i]
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

npv_results = []
budgets = []
pf_cost20r = []
#pf_conf2 = []

#from the sorted dataframe, take the first row, which corresponds to the highest npv portfolio and extract the data needed for the following pictures
finalsol_df = finalsol_df.iloc[0]
portfolio_results = finalsol_df[0]
npv_results_escalar = finalsol_df[1]
npv_results.append(npv_results_escalar)
#npv_results.append(finalsol_df[1])
budgets_escalar = finalsol_df[2]
budgets.append(budgets_escalar)
#budgets.append(finalsol_df[2])
print("portfolio_results: ", portfolio_results)
print("npv_results: ", npv_results)
print("budgets: ", budgets)


#*** execution time
print("Execution time: %s milli-seconds" %((time.time() - start_time)* 1000))

#separate the npv results from the solutions list
#npv_results = [round(x[1][0], 0) for x in solutions]
#separate the portfolio results from the solutions list
#portfolio_results = [x[0] for x in solutions]
#separate the budgets taken from the solutions list (was budgets = [x[2][0] for x in solutions] -> [0] PARA CUANDO SEA SOLO UN BCP
#budgets = [x[2][0] for x in solutions]

#DESACTIVAR ALL THIS SI QUIERES MIRAR TODOS JUNTOS - HASTA PLT(SHOW)
#represent in a scatter plot the results of optimal npv extracted from dataframe finalsol_df vs budgetting confidence policy
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
#solution_portfolios = np.array(portfolio_results)
# plot the square array as a heatmap
#plt.figure(2)
#fig, ax = plt.subplots()
#plt.imshow(solution_portfolios, cmap='binary', interpolation='nearest', vmin=0, vmax=1)
#plt.xlabel("Project", fontsize=12)
#plt.ylabel("Budgetting Confidence Policy", fontsize=14)
#plt.yticks(range(len(budgetting_confidence_policies)), budgetting_confidence_policies, fontsize=12)
#plt.xticks(np.arange(0, nrcandidates, 1), fontsize=12)
#plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)

for i, budget in enumerate(budgets):
    plt.text(nrcandidates + 0.5, i, "${:.2f}".format(budget), ha='left', va='center', fontsize=14)

plt.text(nrcandidates + 2, len(budgetting_confidence_policies) / 2, "Portfolio Budget", ha='center', va='center', rotation=270, fontsize=14)
plt.tight_layout()

#extract the sixth portfolio included in array portfolio_results (RESTORE TO SIX WHEN MORE THAN ONE BCP!!!!!!!!!!!!!)
chosen_portfolio = portfolio_results[0]
#multiply dataframe 0 by the chosen portfolio to reflect the effect of the projects that are chosen
pf_df = df10r * chosen_portfolio # was pf_df = df0 * chosen_portfolio
#sum the rows of the new dataframe to calculate the total cost of the portfolio
pf_cost = pf_df.sum(axis=1)

fig, ax = plt.subplots()
# title of the plot
# ax.set_title('Monte Carlo Simulation of a candidate project')
# Plot the histogram of the monte carlo simulation of the first project
ax.hist(mcs_results2[0][0], bins=200, color='grey', label='Histogram')
# title of the x axis
ax.set_xlabel('Cost in k€')
# Create a twin Axes object that shares the x-axis of the original Axes object
ax2 = ax.twinx()
# Plot the histogram of the monte carlo simulation of the first project in the form of a cumulative distribution function
ax2.hist(mcs_results1[0][0], bins=200, color='black', cumulative=True, histtype='step', density=True, label='Cumulative Distribution')
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
#cm109 = np.full((10, 10), 0.9)
#np.fill_diagonal(cm109, 1)

# Correlation matrix to be used in the next mcs simulation
#cm106 = np.full((10, 10), 0.6)
#np.fill_diagonal(cm106, 1)

# Correlation matrix to be used in the next mcs simulation
#cm103 = np.full((10, 10), 0.3)
#np.fill_diagonal(cm103, 1)



#print(df0)
#print(correlation_matrix0)
# plot the scatter matrix
#pd.plotting.scatter_matrix(df0, alpha=0.2, figsize=(6, 6), diagonal='kde', color='grey', density_kwds={'color': 'grey'})
#plot the scatter matrix of df0 with seaborn pairplot function with grey color and a diagonal with a kde plot
#sns.pairplot(df0, diag_kind="kde", palette="Greys")
# add title and axis labels
#plt.suptitle('Correlation matrix of the MCS results where all projects are fully independent (in k€)')
#plt.xlabel('Projects and cost in k€')
#plt.ylabel('Projects and cost in k€')
#plt.show()



#convert the array of portfolio risks into a dataframe with header each of the correlation levels used
df_portfolio_risk = pd.DataFrame(portfolio_risk)
#transpose the dataframe
df_portfolio_risk = df_portfolio_risk.transpose()
#rename the columns of the dataframe
df_portfolio_risk.rename(columns={0:"0", 1:"0.9", 2:"0.6", 3:"0.3", 4:"random"}, inplace=True)
#current_cols = df_portfolio_risk.columns
print(df_portfolio_risk)

# plot the scatter matrix
pd.plotting.scatter_matrix(df10r, alpha=0.2, figsize=(6, 6), diagonal='kde', color='grey', density_kwds={'color': 'grey'})
# add title and axis labels
plt.suptitle('Correlation matrix of the MCS results where all projs are randomly correlated')
plt.xlabel('Projects and cost in k€')
plt.ylabel('Projects and cost in k€')

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
#plt.figure(12)
#plt.legend().set_visible(False)
#heatmap of the correlation matrix cm10r
#sns.set(font_scale=1.15)
#sns.heatmap(cm10r, annot=True, cmap="Greys")

#plt.show()
