import numpy as np
import cplex  
import math
import os
import sys



from task_rnd_triang_with_interrupts_stdev_new_R2_deterministic import *
from functions_for_simheuristic_v12 import *
from itertools import combinations


# ************** checking upper threshold ****************
nrcandidates = 20
timestamps = []
solutions = []
portfolio_projection = []

# create funtion to be called from another python file
def threshold_calculation(df10r, bestsol_size):
    lo_iterations = 20
    hi_iterations = 100
    # array to store all found solutions
      

    # calculate the costs for each project by utilizing the corresponding functions inside task_rnd_triang_with_interrupts_stdev_new_R2_deterministic.py
    # and store all of them in an array called bdgtperproject_matrix

    # I define a candidate array of size nr candidates with all integer values: ones
    candidatearray = np.ones(nrcandidates)

    # I define an initial array of indexes with all candidates ranging from 0 to nrcandidates-1
    initial_projection_indexes = np.arange(nrcandidates)

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
        
        # add "nrcandidates" variables: the decision on the candidate projects (amount=nrcandidates)
        # Binary variables (B): 0 or 1
        names = ['x'+str(i) for i in range(nrcandidates)]
        p.variables.add(names=names, types=['B']*nrcandidates) 
                                
        # add the constraint(s)    
        p.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=names, val=bdgtperproject_matrix)],
                                senses="L", rhs=[maxbdgt])
        
        # Set the objective to maximize NPV  
        p.objective.set_sense(p.objective.sense.maximize)
    
        # Define the objective function
        p.objective.set_linear([(names[i], npvperproject_matrix[i]) for i in range(nrcandidates)])


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
