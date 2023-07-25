import numpy as np
import time
import cplex  

from task_rnd_triang_with_interrupts_stdev_new_R2_deterministic import *
from functions_for_simheuristic_v12 import *

# ************** checking upper threshold ****************

# array to store all found solutions
solutions = []

nrcandidates = 20
timestamps = []

# calculate the costs for each project by utilizing the corresponding functions inside task_rnd_triang_with_interrupts_stdev_new_R2_deterministic.py
# and store all of them in an array called bdgtperproject_matrix

# I define a candidate array of size nr candidates with all integer values: ones
candidatearray = np.ones(nrcandidates)

# I define an initial array of indexes with all candidates ranging from 0 to nrcandidates-1
initial_projection_indexes = np.arange(nrcandidates)

# first simulation to get all cdfs for cost & benefits before optimization step (may_update: was 1000)
det_results1 = calc_det(candidatearray, 1)

# write the first timestamp and label to the list
timestamps.append(('First deterministic point estimate of budgets and NPV for each project', time.time()))

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

# ************** checking lower threshold ****************

iterations = 100
budgetting_confidence_policies = [0.75]


#initialize an array of budgeted durations that is nrcandidates x len(budgetting_confidence_policies)
budgetedcosts = np.zeros((nrcandidates, len(budgetting_confidence_policies)))
#initialize an array of resulting NPVs that is nrcandidates x len(budgetting_confidence_policies)
resultingNPVs = np.zeros((nrcandidates, len(budgetting_confidence_policies)))

#I define a candidate array of size nr candidates with all ones
candidatearray = np.ones(nrcandidates)
#I define an initial array of indexes with all candidates ranging from 0 to nrcandidates-1
initial_projection_indexes = np.arange(nrcandidates)

#first simulation to get all values for deterministic calculation later via CPlex
mcs_results1 = simulate(candidatearray,iterations)

#print("mcs results1: ", mcs_results1[0])

# mcs_results1[0] corresponds to the project costs and mcs_results1[1] to the project benefits (NPV)
# x_perproj_matrix1 = pointestimate(mcs_results1[0], mcs_results1[1], budgetting_confidence_policies, nrcandidates)
# print ("x_perproj_matrix1: ", x_perproj_matrix1)

# As in this checker we do not use the function pointestimate (which is where we substract NPV-project cost),
# we need to substract it in the next loop: line 123


mcs_results1 = np.array(mcs_results1)
# initialize NPVs and portfolio_budgets array
NPVs = []
portfolio_budgets = []

for i in range(iterations):
    # extract first iteration of mcs_results1 to get the budgeted costs of each project and store it in bdgtperproject_matrix
    bdgtperproject_matrix = np.round(mcs_results1[0, :, i], 2)
    print("bdgtperproject_matrix: ", bdgtperproject_matrix)
    # extract second iteration of mcs_results1 to get the NPV of each project and store it in npvperproject_matrix
    npvperproject_matrix = np.round(mcs_results1[1, :, i], 2)-np.round(mcs_results1[0, :, i], 2)
    print("npvperproject_matrix: ", npvperproject_matrix)
    # define the budget constraint
    maxbdgt = 10800

    # check that bdgtperproject_matrix is a 1D numpy array
    if bdgtperproject_matrix.ndim != 1:
        bdgtperproject_matrix = bdgtperproject_matrix.flatten()

    # check that npvperproject_matrix is a 1D numpy array
    if npvperproject_matrix.ndim != 1:
        npvperproject_matrix = npvperproject_matrix.flatten()

    # multiply each value of solutions array by the corresponding value of bdgtperproject_matrix
    # and store the result in the array budgetedcosts
    for j in range(len(solutions)):
        budgetedcosts[:, j] = bdgtperproject_matrix * solutions[j]
    

    # sum all values of budgetedcosts array and store in a variable named "portfolio budget"
    portfolio_budgets.append(np.sum(budgetedcosts, axis=0))


    # multiply each value of solutions array by the corresponding value of npvperproject_matrix
    # and store the result in the array resultingNPVs
    for j in range(len(solutions)):
        resultingNPVs[:, j] = npvperproject_matrix * solutions[j]

    # sum all values of budgetedcosts array and store in a variable named "portfolio budget"
    NPVs.append(np.sum(resultingNPVs, axis=0))


# print the array NPVs
print("NPVs: ", NPVs)

# sum all values of NPVs and store in a variable named "portfolio NPV"
portfolio_NPV = np.sum(NPVs, axis=0)

# calculate the average value of the solutions
average_Portfolio_Budget = np.mean(portfolio_budgets)
print("Array of Pf_Budgets of the iterated Solutions:", portfolio_budgets)
print("Average value of Pf_Budget achieved: ", average_Portfolio_Budget)


# calculate the average value of the solutions
averageNPV = np.mean(NPVs)
print("Array of NPVs of the iterated Solutions:", NPVs)
print("Average value of NPV's achieved: ", averageNPV)



