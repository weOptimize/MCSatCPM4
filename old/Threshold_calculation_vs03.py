import numpy as np
import time
import cplex  

from task_rnd_triang_with_interrupts_stdev_new_R2_deterministic import *
from functions_for_simheuristic_v12 import *

# ************** checking upper threshold ****************

# create funtion to be called from another python file
def threshold_calculation(df10r):
    # array to store all found solutions
    solutions = []
    portfolio_projection = []

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

    # convert the list solutions into a new one called portfolio_projection that includes only integer values (0 or 1)
    portfolio_projection = [int(i) for i in solutions[0]]

    # convert portfolio_projection array into a binary array
    # portfolio_projection = portfolio_projection.astype(int)
    print("portfolio_projection: ", portfolio_projection)

    # I want to reuse code that analyzes a HoF, but now I want it to analyze only one solution, 
    # so I create a list with only one element
    projectselection = []
    projectselection.append(portfolio_projection)


    # ************** checking lower threshold ****************

    iterations = 100
    budgetting_confidence_policies = [0.75]

    print ("************ Checking Lower Threshold **********")
    #second simulation to get all cdfs for cost & benefits after optimization step (may_update: was 1000)
    mcs_results2 = simulate(portfolio_projection,iterations)

    # calculate the amount of projects in "portfolio_projection"
    projected_candidates = sum(portfolio_projection)

    # store the positions of the chosen projects in the portfolio_projection array, starting with 0 (as i+1 for if if starting with 1)
    zipped_projection_indexes = [i for i, x in enumerate(portfolio_projection) if x == 1]

    # mcs_results2[0] corresponds to the project costs and mcs_results2[1] to the project benefits (NPV)
    x_perproj_matrix2 = pointestimate(mcs_results2[0], mcs_results2[1], budgetting_confidence_policies, projected_candidates)
    # print ("x_perproj_matrix2: ", x_perproj_matrix2)

    # we assume correlations at the cost side, not at the benefits side (conservative approach)
    # update x_perproj_matrix2 with the correlation effect registered inside df20r
    # print("x_perproj_matrix2: ", x_perproj_matrix2)
    # separate the budget and npv results from the x_perproj_matrix
    bdgtperproject_matrix = x_perproj_matrix2[0]
    npvperproject_matrix = x_perproj_matrix2[1]
    # print(type(bdgtperproject_matrix))
    # print(type(npvperproject_matrix))
    bdgtperproject_matrix = np.squeeze(bdgtperproject_matrix)
    npvperproject_matrix = np.squeeze(npvperproject_matrix)

    # remove all data that has zeroes from bdgtperproject_matrix and npvperproject_matrix
    # bdgtperproject_matrix = bdgtperproject_matrix[np.nonzero(bdgtperproject_matrix.flatten())]
    # npvperproject_matrix = npvperproject_matrix[np.nonzero(npvperproject_matrix.flatten())]

    # print("bdgtperproject_matrix: ", bdgtperproject_matrix)
    # print("npvperproject_matrix: ", npvperproject_matrix)
    print("size of bdgtperproject_matrix", len(bdgtperproject_matrix))
    print("size of npvperproject_matrix", len(npvperproject_matrix))
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
    # enlarge the dataframe to the size of iterations
    widened_df20r = widened_df20r.reindex(range(iterations))
    # fill the dataframe with zeroes
    widened_df20r.iloc[:, :] = 0

    df20r = correlatedMCS(mcs_results2, iterations, projected_candidates, zipped_projection_indexes)
    # print("df20r: ", df20r)

    # pick in order the values from bdgtperproject_matrix and npvperproject_matrix and store them in widened_bdgtperproject_matrix and widened_npvperproject_matrix
    # The location of the values to be picked is available in zipped_projection_indexes
    j=0
    for i in range(nrcandidates):
        if i in zipped_projection_indexes:
            widened_bdgtperproject_matrix [i] = round(bdgtperproject_matrix [j],3)
            widened_npvperproject_matrix [i] = round(npvperproject_matrix [j],3)
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
            for j in range(iterations):
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
        #array storing the portfolio risk not to exceed 10.800 Mio.€, as per-one risk units
        pf_conf2[i] = 1-count/iterations

    # create a dataframe with the results
    finalsol_df = pd.DataFrame({'Portfolio': projectselection, 'Portfolio NPV': npv_results, 'Portfolio Budget': budgets, 'Portfolio confidence': pf_conf2})
    # order the dataframe by the portfolio npv, starting with the highest npv
    finalsol_df = finalsol_df.sort_values(by=['Portfolio NPV'], ascending=False)
    print ("Final Solution: ", finalsol_df)

    # write the fifth timestamp and label to the list. Calculation FINALIZED
    timestamps.append(('Application of correlation effect to final options', time.time()))

    segments = [0] * (len(timestamps)-1)

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
    print ("Indexes of selected projects at deterministic portfolio: ", zipped_projection_indexes)
    print("portfolio_results: ", portfolio_results)
    print("npv_results: ", npv_results)
    print("budgets: ", budgets)

    # from the projects at the selected portfolio, extract the costs and benefits of each project
    # and store them in a matrix, together with the project indexes
    
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
     
     


