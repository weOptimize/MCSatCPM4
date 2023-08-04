import numpy as np
import cplex  
import math
import os
import sys



from task_rnd_triang_with_interrupts_stdev_new_R2_deterministic import *
from functions_for_simheuristic_v12 import *
from itertools import combinations
from functions_for_simheuristic_v12 import *


# ************** checking upper threshold ****************
initcandidates = 20
timestamps = []
solutions = []
portfolio_projection = []

# create funtion to be called from another python file
def threshold_calculation(df10r, bestsol_size):
    lo_iterations = 20 #was 20
    hi_iterations = 100 #was 100
    # array to store all found solutions
      

    # calculate the costs for each project by utilizing the corresponding functions inside task_rnd_triang_with_interrupts_stdev_new_R2_deterministic.py
    # and store all of them in an array called bdgtperproject_matrix

    # I define a candidate array of size nr candidates with all integer values: ones
    candidatearray = np.ones(initcandidates)

    # I define an initial array of indexes with all candidates ranging from 0 to initcandidates-1
    initial_projection_indexes = np.arange(initcandidates)

    # first simulation to get all cdfs for cost & benefits before optimization step (may_update: was 1000)
    det_results1 = calc_det(candidatearray, 1)

    # extract first column of the matrix to get the budgeted costs of each project and store it in bdgtperproject_matrix
    bdgtperproject_matrix = np.round(det_results1[0], 2)
    print("bdgtperproject_matrix: ", bdgtperproject_matrix)
    # extract second column of the matrix to get the NPV of each project and store it in npvperproject_matrix
    npvperproject_matrix = np.round(det_results1[1], 2)
    print("npvperproject_matrix: ", npvperproject_matrix)
    # define the budget constraint
    maxbdgt = 10800

    # check that bdgtperproject_matrix is a 1D numpy array
    if bdgtperproject_matrix.ndim != 1:
        bdgtperproject_matrix = bdgtperproject_matrix.flatten()

    # check that npvperproject_matrix is a 1D numpy array
    if npvperproject_matrix.ndim != 1:
        npvperproject_matrix = npvperproject_matrix.flatten()
        

    # pass values of bdgtperproject_matrix and npvperproject_matrix to CPLEX to perform a maximization of the NPV
    # and return the optimal solution (the optimal portfolio) and the optimal NPV

    def setproblemdata(p):
        # minimize risk while keeping the return constant
        p.objective.set_sense(p.objective.sense.minimize)  
        
        # add "initcandidates" variables: the decision on the candidate projects (amount=initcandidates)
        # Binary variables (B): 0 or 1
        names = ['x'+str(i) for i in range(initcandidates)]
        p.variables.add(names=names, types=['B']*initcandidates) 
                                
        # add the constraint(s)    
        p.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=names, val=bdgtperproject_matrix)],
                                senses="L", rhs=[maxbdgt])
        
        # Set the objective to maximize NPV  
        p.objective.set_sense(p.objective.sense.maximize)
    
        # Define the objective function
        p.objective.set_linear([(names[i], npvperproject_matrix[i]) for i in range(initcandidates)])


    # Define the problem
    p = cplex.Cplex()
    setproblemdata(p)
    # Solve the problem
    p.solve()


    # Print the results
    print("Optimal solution:")
    print(p.solution.get_values())
    print("Optimal NPV: %.2f" % p.solution.get_objective_value())

    # assign the result from projectselection to the variable solutions
    solutions.append(p.solution.get_values())

    # convert the list solutions into a new one called deterministic_portfolio that includes only integer values (0 or 1)
    deterministic_portfolio = [int(i) for i in solutions[0]]

    # convert deterministic_portfolio array into a binary array
    # deterministic_portfolio = deterministic_portfolio.astype(int)
    print("deterministic_portfolio: ", deterministic_portfolio)

    # I want to reuse code that analyzes a HoF, but now I want it to analyze only one solution, 
    # so I create a list with only one element
    projectselection = []
    projectselection.append(deterministic_portfolio)

    print ("************ Checking Lower Threshold **********")
    simulatescenario(df10r, deterministic_portfolio, projectselection, hi_iterations)

    # take the indexes of the projects in the deterministic portfolio and store them in a list
    zipped_projection_indexes = [i for i, x in enumerate(deterministic_portfolio) if x == 1]

    # set the print options to suppress scientific notation
    np.set_printoptions(suppress=True)

    # create an array with the indexes of the projects in the portfolio
    indexes_array = np.zeros((len(zipped_projection_indexes)))
    for i in range(len(zipped_projection_indexes)):
        indexes_array[i] = zipped_projection_indexes[i]

    # convert the array content into integer values
    indexes_array = indexes_array.astype(int)

    # create an array  with the costs of the projects in the portfolio
    costs_array = np.zeros((len(zipped_projection_indexes)))
    for i in range(len(zipped_projection_indexes)):
        costs_array[i] = round(bdgtperproject_matrix[i],3)

    # create an array  with the npv of the projects in the portfolio
    npv_array = np.zeros((len(zipped_projection_indexes)))
    for i in range(len(zipped_projection_indexes)):
        npv_array[i] = round(npvperproject_matrix[i],3)

    #create an array with the cost/npv ratio of the projects in the portfolio
    ratio_cost_npv_array = np.zeros((len(zipped_projection_indexes)))
    for i in range(len(zipped_projection_indexes)):
        ratio_cost_npv_array[i] = round(bdgtperproject_matrix[i]/npvperproject_matrix[i],2)

    # create a matrix with the indexes, costs, npv and ratio of the projects in the deterministic portfolio
    deterministic_matrix = np.zeros((len(zipped_projection_indexes),4))
    for i in range(len(zipped_projection_indexes)):
        deterministic_matrix[i] = [indexes_array[i], costs_array[i], npv_array[i], ratio_cost_npv_array[i]]
    print("deterministic_matrix: ", deterministic_matrix)

    # reorder all data in the matrix by the ratio cost/npv from lowest to highest
    reordered_matrix = deterministic_matrix[deterministic_matrix[:,3].argsort()]
    print("reordered_matrix: ", reordered_matrix)


    # generate as many reduced deterministic portfolios as projects in the deterministic portfolio
    # and each time remove the project with the highest ratio cost/npv, and then remove another one
    # keeping the previous without removal (in other words, each portfolio generated has the same amount)

# calculate how many combinations are when there are as many elements as the number of projects
# in the deterministic portfolio and the groups have the size of bestsol_size
    n = len(zipped_projection_indexes) - bestsol_size
    n_combinations = 0
    i=0
    reduced_deterministic_indexes = list(combinations(zipped_projection_indexes, bestsol_size))
    n_combinations = int(math.factorial(len(zipped_projection_indexes)) /
                        (math.factorial(bestsol_size) * math.factorial(len(zipped_projection_indexes) - bestsol_size)))
    for i in range(1, n):
        k = bestsol_size + i
        n_combinations += int(math.factorial(len(zipped_projection_indexes)) / 
                            (math.factorial(k) * math.factorial(len(zipped_projection_indexes) - k)))
        # add the new combinations to the existing list
        reduced_deterministic_indexes.extend(list(combinations(zipped_projection_indexes, k)))

    # create a list to store the reduced deterministic portfolios sized as the number of subgroup combinations
    reduced_deterministic_portfolios = [0] * n_combinations

    for i in range(len(reduced_deterministic_portfolios)):
        # copy the deterministic portfolio inside the i-th element of the reduced_deterministic_portfolios list
        reduced_deterministic_portfolios[i] = deterministic_portfolio.copy()
        for j in range(n):
            # multiply by zero the elements that corresponds to the positions of zipped projection indexes inside
            # the corresponding row at reduced_deterministic_indexes
            # identify what projection indexes are missing
            missing_projection_indexes = [x for x in zipped_projection_indexes if x not in reduced_deterministic_indexes[i]]
            for k in range(len(missing_projection_indexes)):
                reduced_deterministic_portfolios[i][missing_projection_indexes[k]] = 0
        
      
    # print("reduced_deterministic_portfolios: ", reduced_deterministic_portfolios)
    
    # then use function simulatescenario of each reduced deterministic portfolioand store the results in a list
    # the list must include the index of the removed project, the cost of the portfolio, the npv of
    # the portfolio, and the confidence level of each reduced deterministic portfolio


    # create a list to store the results of each reduced deterministic portfolio
    reduced_deterministic_portfolios_results = []
    count_Hiconf = 0
    for i in range(len(reduced_deterministic_portfolios)):
        # reset array projectselection so that I can include the reduced deterministic portfolio
        projectselection = []
        projectselection.append(reduced_deterministic_portfolios[i])
        reduced_deterministic_portfolios_results.append(simulatescenario(df10r, reduced_deterministic_portfolios[i], projectselection, lo_iterations))
        # count the number of results that have a confidence level higher than 0.75
        if reduced_deterministic_portfolios_results[i][3][0] > 0.75:
            count_Hiconf += 1


    # print only the first 5 elements from reduced_deterministic_portfolios_results and last 5 results
    # and inform the total amount of results


    print ("Total number of High Confidence (>65%) results: ", count_Hiconf)
    print("reduced_deterministic_portfolios_results: ")
    for i in range(5):
        print(reduced_deterministic_portfolios_results[i])
    print("...")
    for i in range(len(reduced_deterministic_portfolios_results)-5, len(reduced_deterministic_portfolios_results)):
        print(reduced_deterministic_portfolios_results[i])
    print("Total number of results: ", len(reduced_deterministic_portfolios_results))
    print("reduced_deterministic_portfolios_results: ", reduced_deterministic_portfolios_results)
    


    # extract all results that resulted in a confidence level higher than 0.75
    # and iterate over such results with iter = 200 iterations
    # and then store the results in a list in descending order
    reduced_deterministic_portfolios_results_Hi_confidence = []
    for i in range(len(reduced_deterministic_portfolios_results)):
        if reduced_deterministic_portfolios_results[i][3][0] > 0.75:
            reduced_deterministic_portfolios_results_Hi_confidence.append(simulatescenario(df10r, reduced_deterministic_portfolios[i], projectselection, hi_iterations))
            print (len(reduced_deterministic_portfolios_results_Hi_confidence))

    #sort in descending order of Net Present Value
    reduced_deterministic_portfolios_results_Hi_confidence.sort(key=lambda x: x[2], reverse=True)
    print("reduced_deterministic_portfolios_results_Hi_confidence: ", reduced_deterministic_portfolios_results_Hi_confidence)

    #extract the first element of the list that meets a criteria of confidence level higher than 0.90 and print it
    for i in range(len(reduced_deterministic_portfolios_results_Hi_confidence)):
        if reduced_deterministic_portfolios_results_Hi_confidence[i][3][0] > 0.90:
            print("reduced_deterministic_portfolios_results_Hi_confidence: ", reduced_deterministic_portfolios_results_Hi_confidence[i])
            break