#This file includes the function that returns the survival value for a given budgetting confidence policy.
#It is called from the main file
import math
import random
import numpy as np
import pandas as pd
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
from operator import itemgetter
from pandas_ods_reader import read_ods
from copulas.multivariate import GaussianMultivariate
from scipy.stats import rv_continuous, rv_histogram, norm, uniform, multivariate_normal, beta
from fitter import Fitter, get_common_distributions, get_distributions


#import created scripts:
from task_rnd_triang_with_interrupts_stdev_new_R2 import *
from task_rnd_triang_with_interrupts_stdev_new_R2_deterministic import *

#I define the number of candidates to be considered
initcandidates = 20
nr_confidence_policies = 1
mcs_costs = []
mcs_PV = []
maxbdgt = 3800
#initialize matrices to store bdgt and npv
bdgtperproject_matrix = np.zeros((initcandidates, nr_confidence_policies))
PVperproject_matrix = np.zeros((initcandidates, nr_confidence_policies))


#defining the function that calculates the total budget of a portfolio of projects
def portfolio_totalbudget(portfolio,bdgtperproject):
    totalbudget_portfolio = 0
    #totalbudget_npv = 0
    for i in range(initcandidates):
        if portfolio[i] == 1:
            totalbudget_portfolio += bdgtperproject[i]
            #totalbudget_npv += npvperproject[i]
    #return totalbudget_portfolio, totalbudget_npv
    return totalbudget_portfolio


#define the function that returns the survival value for a given budgetting confidence policy
def survival_value_extractor(sim_costs, budgetting_confidence_policy, iterations):
    #calculate the cumulative sum of the values of the histogram
	valuesplus, base = np.histogram(sim_costs, bins=iterations) #it returns as many values as specified in bins valuesplus are frequencies, base the x-axis limits for the bins 
	cumulativeplus = np.cumsum(valuesplus)
	survivalvalues = 100*(len(sim_costs)-cumulativeplus)/len(sim_costs)
	#return index of item from survivalvalues that is closest to "1-budgetting_confidence_policy" typ.20%
	index = (np.abs(survivalvalues-100*(1-budgetting_confidence_policy))).argmin()
	#return value at base (which is indeed the durations that correspond to survival level) that matches the index
	budgetedduration = np.round(base[index],2)
	return budgetedduration
    

#define the function that returns the expected value for a given budgetting confidence policy
def expected_value_extractor(sim_PV, iterations):
    #calculate the cumulative sum of the values of the histogram
	valuesplus, base = np.histogram(sim_PV, bins=iterations) #it returns as many values as specified in bins valuesplus are frequencies, base the x-axis limits for the bins 
	cumulativeplus = np.cumsum(valuesplus)
	survivalvalues = 100*(len(sim_PV)-cumulativeplus)/len(sim_PV)
	#return index of item from survivalvalues that is closest to "1-budgetting_confidence_policy" typ.20%. Here I place 50% because I want to use the median=avg=E()
	index = (np.abs(survivalvalues-100*(1-.5))).argmin()
	#return value at base (which is indeed the durations that correspond to survival level) that matches the index
	budgetedPV = np.round(base[index],2)
	return budgetedPV



def simulate(arrayforsim, iterat):
    #initialize the arrays that will store the results of the MonteCarlo Simulation
    mcs_costs = []
    mcs_PV = []
    for i in range(len(arrayforsim)):        
        #if the value i is 1, then the simulation is performed
        if arrayforsim[i] == 1:
            # open ten different ODS files and store the results in a list after computing the CPM and MCS 
            # (restore to only last line if old version)
            if i < 9:
                filename = "RND_Schedules/data_wb0" + str(i+1) + ".ods"
            else:
                filename = "RND_Schedules/data_wb" + str(i+1) + ".ods"  
            #print(filename)
            mydata = read_ods(filename, 1)
            # open ten different ODS files and store the results in a list after computing the CPM and MCS 
            # (restore to only last line if old version)
            if i < 9:
                filename = "RND_Schedules/riskreg_0" + str(i+1) + ".ods"
            else:
                filename = "RND_Schedules/riskreg_" + str(i+1) + ".ods"
            #print(filename)
            myriskreg = read_ods(filename, 1) # was myriskreg = read_ods(filename, "Sheet1")

            #compute MonteCarlo Simulation and store the results in an array called "sim_costs"
            sim_costs = MCS_CPM_RR(mydata, myriskreg, iterat)
            cashflows = []
            # open the file that contains the expected cash flows, and extract the ones for the project i (located in row i)
            with open('RND_Schedules/expected_cash_flows.txt') as f:
                # read all the lines in the file as a list
                lines = f.readlines()
                # get the line at index i (assuming i is already defined)
                line = lines[i]
                # split the line by whitespace and convert each element to a float
                cashflows = list(map(float, line.split()))

            # compute MonteCarlo Simulation and store the results in an array called "sim1_PV"
            #print(cashflows)
            sim_PV = MCS_PV(cashflows, iterat)
            #print(sim_PV)
            
            #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the cost at each iteration
            mcs_costs.append(sim_costs)
            mcs_PV.append(sim_PV)
            #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the PV at each iteration
            #mcs_npvs1.append(sim1_PV)
            #compute the median of the PV results
            median_PV = expected_value_extractor(sim_PV, iterat)
        else:
            # if the value i is 0, then the simulation is not performed and "nothing is done" (was "the appended results an array full of zeros")
            # mcs_PV.append([0.0])   
            # mcs_costs.append(np.zeros(iterat))
            # do nothing and go to the next iteration
            pass
         
            

    # print ("mcs_costs", mcs_costs)
    # print ("mcs_PV", mcs_PV)
    return(mcs_costs, mcs_PV)

# compute the median of the PV results
def pointestimate(mcs_costs, mcs_PV, budgetting_confidence_policies, numberofprojects):
    #initialize the arrays that will store the point estimates with size nr of projects x nr of budgetting confidence policies
    bdgtperproject_matrix = np.zeros((numberofprojects, len(budgetting_confidence_policies)))
    PVperproject_matrix = np.zeros((numberofprojects, len(budgetting_confidence_policies)))
    for i in range(numberofprojects):
        median_PV = round(expected_value_extractor(mcs_PV[i], len(mcs_PV[i])),0)
        for j in range(len(budgetting_confidence_policies)):
            budgetting_confidence_policy = budgetting_confidence_policies[j]
            #extract the survival value from the array sim_duration that corresponds to the budgetting confidence policy
            survival_value = survival_value_extractor(mcs_costs[i], budgetting_confidence_policy, len(mcs_costs[i]))
            #store the first survival value in an array where the columns correspond to the budgetting confidence policies and the rows correspond to the projects
            bdgtperproject_matrix[i][j]=survival_value
            # npvperproject_matrix[i][j]=median_npv-survival_value #(was npvperproject_matrix[i][j]=median_npv-survival_value)
            PVperproject_matrix[i][j]=median_PV  #revert if wanting to get NPV instead of PV
    # print ("bdgtperproject_matrix", bdgtperproject_matrix)
    # print ("npvperproject_matrix", npvperproject_matrix)
    return(bdgtperproject_matrix, PVperproject_matrix)

# modify MCS results to reflect the correlation matrix  
def correlatedMCS(mcs_results, iterat, nrcandidates, projection_indexes):  
    #check the parameters of beta distribution for each of the mcs_results  
    betaparams = []  
    for i in range(nrcandidates):  
        f = Fitter(mcs_results[0][i], distributions=['beta'])
        f.fit(progress=False)
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
  
    # print("betaparams: ")
    # print(betaparams)  
  
    # copy the array with all MCS results  
    df0 = pd.DataFrame(data=mcs_results[0]).T  
    col_names = ["P{:02d}".format(i+1) for i in range(nrcandidates)]  
    df0.rename(columns=dict(enumerate(col_names)), inplace=True)  
    correlation_matrix0 = df0.corr()  
  
    # *********Correlation matrix with random values between 0 and 1, but positive semidefinite***************  
    # Set the seed value for the random number generator    
    seed_value = 1005    
    np.random.seed(seed_value)  
    # Generate a random symmetric matrix  
    A = np.random.rand(initcandidates, initcandidates)  
    A = (A + A.T) / 2  
    # Compute the eigenvalues and eigenvectors of the matrix  
    eigenvalues, eigenvectors = np.linalg.eigh(A)  
    # Ensure the eigenvalues are positive  
    eigenvalues = np.abs(eigenvalues)  
    # Normalize the eigenvalues so that their sum is equal to nrcandidates  
    eigenvalues = eigenvalues / eigenvalues.sum() * initcandidates  
    # Compute the covariance matrix. Forcing positive values, as long as negative correlations are not usual in reality of projects  
    cm10r = np.abs(eigenvectors.dot(np.diag(eigenvalues)).dot(eigenvectors.T))  
    # Ensure the diagonals are equal to 1  
    for i in range(initcandidates):  
        cm10r[i, i] = 1  
    # print('cm10r BEFORE:')
   #  print(cm10r)
    
    # if the sum of the values inside projection_indexes is the same as the number of candidates, then we do not change the correlation matrix
    if len(projection_indexes) == initcandidates:
        cm10r = cm10r
    # if the sum of the values inside projection_indexes is not the same as the number of candidates, then we change the correlation matrix
    else:
        # we change the correlation matrix by setting the correlation between the candidates that are not selected to 0
        # for each time we find a zero, we trim the whole column adn row of the correlation matrix corresponding to that candidate
        i = 0
        j = 0
        for i in range(initcandidates):
            if i not in projection_indexes:
                cm10r = np.delete(cm10r, i-j, 0)
                cm10r = np.delete(cm10r, i-j, 1)
                j+=1
    # print('cm10r AFTER:')
    # print(cm10r)

# ***************** ONLY ACTIVE WHEN TESTING NO-CORRELATION VERSION *****************
    # set the correlation matrix to the identity matrix (no correlation)
    # cm10r = np.identity(initcandidates)

    if cm10r.shape[0] == initcandidates:
        #make sure no legend appears in the next plot
        plt.figure(12)
        #plt.legend().set_visible(False)
        #heatmap of the correlation matrix cm10r
        sns.set(font_scale=1.15)
        sns.heatmap(cm10r, annot=True, cmap="Greys")

    #initialize dataframe df10r with size nrcandidates x iterations  
    df10r = pd.DataFrame(np.zeros((iterat, nrcandidates)))  
    # step 1: draw random variates from a multivariate normal distribution   
    # with the targeted correlation structure  
    r0 = [0] * cm10r.shape[0]                       # create vector r with as many zeros as correlation matrix has variables (row or columns)  
    mv_norm = multivariate_normal(mean=r0, cov=cm10r)    # means = vector of zeros; cov = targeted corr matrix  
    rand_Nmv = mv_norm.rvs(iterat)                               # draw N random variates  
    # step 2: convert the r * N multivariate variates to scores   
    rand_U = norm.cdf(rand_Nmv)   # use its cdf to generate N scores (probabilities between 0 and 1) from the multinormal random variates  
    # step 3: instantiate the nrcandidates marginal distributions   
    d_list = []  
    for i in range(nrcandidates):  
        d = beta(a[i], b[i], loc[i], scale[i])  
        d_list.append(d)  
    # draw N random variates for each of the nrcandidates marginal distributions  
    # WITHOUT applying a copula
    # do it only for the ones different from 0  
    rand_list = [d.rvs(iterat) for d in d_list]  
    # rand_list = [d.rvs(iterat) for i, d in enumerate(d_list) if i not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]  
    # initial correlation structure before applying a copula  
    c_before = np.corrcoef(rand_list)  
    # step 4: draw N random variates for each of the nrcandidates marginal distributions  
    # and use as inputs the correlated uniform scores we have generated in step 2  
    rand_list = [d.ppf(rand_U[:, i]) for i, d in enumerate(d_list)]  
    # final correlation structure after applying a copula  
    c_after = np.corrcoef(rand_list)  
    #print("Correlation matrix before applying a copula:")  
    #print(c_before)  
    #print("Correlation matrix after applying a copula:")  
    #print(c_after)  
    # step 5: store the N random variates in the dataframe  
    for i in range(nrcandidates):  
        df10r[i] = rand_list[i]  
    col_names = ["P{:02d}".format(i+1) for i in range(nrcandidates)]  
    df10r.rename(columns=dict(enumerate(col_names)), inplace=True)  
    correlation_matrix1 = df10r.corr()  
    return df10r  

def NONcorrelatedMCS(mcs_results, iterat, nrcandidates, projection_indexes):  
    # pass non-correlated MCS results to dataframe  
    df0 = pd.DataFrame(data=mcs_results[0]).T  
    col_names = ["P{:02d}".format(i+1) for i in range(nrcandidates)]  
    df0.rename(columns=dict(enumerate(col_names)), inplace=True)  
    correlation_matrix0 = df0.corr()
    return df0

def calc_det(arrayforsim, iterat):
    #initialize the arrays that will store the results of the MonteCarlo Simulation
    det_costs = []
    det_PV = []
    for i in range(len(arrayforsim)):        
        #if the value i is 1, then the simulation is performed
        if arrayforsim[i] == 1:
            # open ten different ODS files and store the results in a list after computing the CPM and MCS 
            # (restore to only last line if old version)
            if i < 9:
                filename = "RND_Schedules/data_wb0" + str(i+1) + ".ods"
            else:
                filename = "RND_Schedules/data_wb" + str(i+1) + ".ods"  
            #print(filename)
            mydata = read_ods(filename, 1)
            # open ten different ODS files and store the results in a list after computing the CPM and MCS 
            # (restore to only last line if old version)
            if i < 9:
                filename = "RND_Schedules/riskreg_0" + str(i+1) + ".ods"
            else:
                filename = "RND_Schedules/riskreg_" + str(i+1) + ".ods"
            #print(filename)
            myriskreg = read_ods(filename, 1) # was myriskreg = read_ods(filename, "Sheet1")

            #compute MonteCarlo Simulation and store the results in an array called "sim1_costs"
            sim_costs = MCS_CPM_RRdet(mydata, myriskreg, iterat)
            cashflows = []
            # open the file that contains the expected cash flows, and extract the ones for the project i (located in row i)
            with open('RND_Schedules/expected_cash_flows.txt') as f:
                # read all the lines in the file as a list
                lines = f.readlines()
                # get the line at index i (assuming i is already defined)
                line = lines[i]
                # split the line by whitespace and convert each element to a float
                cashflows = list(map(float, line.split()))

            # compute MonteCarlo Simulation and store the results in an array called "sim1_PV", also 
            sim_PV = MCS_PVdet(cashflows, iterat)
            # print(sim_NPV)
            # substract sim_costs from all the values inside the array
            for j in range(len(sim_PV)):
                sim_PV[j] = sim_PV[j] #activate below if looking for npv instead of pv
                #sim_PV[j] = sim_PV[j] - sim_costs[j]
            #print(sim_PV)
            
            #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the cost at each iteration
            det_costs.append(sim_costs)
            det_PV.append(sim_PV)
            #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the PV at each iteration
            #mcs_npvs1.append(sim1_PV)
            #compute the median of the PV results
        else:
            # if the value i is 0, then the simulation is not performed and "nothing is done" (was "the appended results an array full of zeros")
            # mcs_PV.append([0.0])   
            # mcs_costs.append(np.zeros(iterat))
            # do nothing and go to the next iteration
            pass

    return(det_costs, det_PV)

def calc_det_withReserves(arrayforsim, iterat):
    #initialize the arrays that will store the results of the MonteCarlo Simulation
    det_costs = []
    det_PV = []
    for i in range(len(arrayforsim)):        
        #if the value i is 1, then the simulation is performed
        if arrayforsim[i] == 1:
            # open ten different ODS files and store the results in a list after computing the CPM and MCS 
            # (restore to only last line if old version)
            if i < 9:
                filename = "RND_Schedules/data_wb0" + str(i+1) + ".ods"
            else:
                filename = "RND_Schedules/data_wb" + str(i+1) + ".ods"  
            #print(filename)
            mydata = read_ods(filename, 1)
            # open ten different ODS files and store the results in a list after computing the CPM and MCS 
            # (restore to only last line if old version)
            if i < 9:
                filename = "RND_Schedules/riskreg_0" + str(i+1) + ".ods"
            else:
                filename = "RND_Schedules/riskreg_" + str(i+1) + ".ods"
            #print(filename)
            myriskreg = read_ods(filename, 1) # was myriskreg = read_ods(filename, "Sheet1")

            #compute MonteCarlo Simulation and store the results in an array called "sim1_costs"
            sim_costs = MCS_CPM_RRdet_withReserves(mydata, myriskreg, iterat)
            cashflows = []
            # open the file that contains the expected cash flows, and extract the ones for the project i (located in row i)
            with open('RND_Schedules/expected_cash_flows.txt') as f:
                # read all the lines in the file as a list
                lines = f.readlines()
                # get the line at index i (assuming i is already defined)
                line = lines[i]
                # split the line by whitespace and convert each element to a float
                cashflows = list(map(float, line.split()))

            # compute MonteCarlo Simulation and store the results in an array called "sim1_PV", also 
            sim_PV = MCS_PVdet(cashflows, iterat)
            # print(sim_PV)
            # substract sim_costs from all the values inside the array
            for j in range(len(sim_PV)):
                sim_PV[j] = sim_PV[j] #activate below if looking for npv instead of pv
                #sim_NPV[j] = sim_NPV[j] - sim_costs[j]
            #print(sim_NPV)
            
            #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the cost at each iteration
            det_costs.append(sim_costs)
            det_PV.append(sim_PV)
            #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the PV at each iteration
            #mcs_npvs1.append(sim1_PV)
            #compute the median of the PV results
        else:
            # if the value i is 0, then the simulation is not performed and "nothing is done" (was "the appended results an array full of zeros")
            # mcs_PV.append([0.0])   
            # mcs_costs.append(np.zeros(iterat))
            # do nothing and go to the next iteration
            pass

    return(det_costs, det_PV)

def simulatescenario0(df10r, portfolio_projection, projectselection, iter):
    iterations = iter
    maxbdgt = 10800
    budgetting_confidence_policies = [0.75]


    #second simulation to get all cdfs for cost & benefits after optimization step
    mcs_results2 = simulate(portfolio_projection,iterations)

    # calculate the amount of projects in "portfolio_projection"
    projected_candidates = sum(portfolio_projection)

    # store the positions of the chosen projects in the portfolio_projection array, starting with 0 (as i+1 for if if starting with 1)
    zipped_projection_indexes = [i for i, x in enumerate(portfolio_projection) if x == 1]

    # mcs_results2[0] corresponds to the project costs and mcs_results2[1] to the project benefits (PV)
    x_perproj_matrix2 = pointestimate(mcs_results2[0], mcs_results2[1], budgetting_confidence_policies, projected_candidates)
    # print ("x_perproj_matrix2: ", x_perproj_matrix2)

    # we assume correlations at the cost side, not at the benefits side (conservative approach)
    # update x_perproj_matrix2 with the correlation effect registered inside df20r
    # print("x_perproj_matrix2: ", x_perproj_matrix2)
    # separate the budget and npv results from the x_perproj_matrix
    bdgtperproject_matrix = x_perproj_matrix2[0]
    PVperproject_matrix = x_perproj_matrix2[1]
    # print(type(bdgtperproject_matrix))
    # print(type(npvperproject_matrix))
    bdgtperproject_matrix = np.squeeze(bdgtperproject_matrix)
    PVperproject_matrix = np.squeeze(PVperproject_matrix)

    # remove all data that has zeroes from bdgtperproject_matrix and npvperproject_matrix
    # bdgtperproject_matrix = bdgtperproject_matrix[np.nonzero(bdgtperproject_matrix.flatten())]
    # npvperproject_matrix = npvperproject_matrix[np.nonzero(npvperproject_matrix.flatten())]

    # print("bdgtperproject_matrix: ", bdgtperproject_matrix)
    # print("npvperproject_matrix: ", npvperproject_matrix)
    # print("size of bdgtperproject_matrix", len(bdgtperproject_matrix))
    # print("size of npvperproject_matrix", len(npvperproject_matrix))
    # print("size of mcs_results2", len(mcs_results2))

    # print("mcs_results2 (input para correlacionar): ", mcs_results2)

    # for each of the options obtained in projectselection, calculate the total portfolio npv and the portfolio budget based on the information from x_perproj_matrix
    PV_results = [0] * len(projectselection) # as many as len(projectselection) because we have one npv per item in HoF
    budgets = [0] * len(projectselection)
    pf_conf2 = [0] * len(projectselection)
    widened_bdgtperproject_matrix = [0] * initcandidates # as many as initial amount of project candidates
    widened_PVperproject_matrix = [0] * initcandidates
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
    for i in range(initcandidates):
        if i in zipped_projection_indexes:
            widened_bdgtperproject_matrix [i] = round(bdgtperproject_matrix [j],3)
            widened_PVperproject_matrix [i] = round(PVperproject_matrix [j],3)
            j+=1
        else:
            pass
    # print("widened_bdgtperproject_matrix: ", widened_bdgtperproject_matrix)
    # print("widened_npvperproject_matrix: ", widened_npvperproject_matrix)

    # pick in order the values from df20r and store them in widened_df20r (to be used in the next step)
    i=0
    j=0
    k=0
    for i in range(initcandidates):
        if i in zipped_projection_indexes:
            for j in range(iterations):
                widened_df20r.loc[j, widened_df20r.columns[i]] = df20r.loc[j, df20r.columns[k]]
            k += 1
        else:
            pass

    # print("widened_df20r: ", widened_df20r)

    for i in range(len(projectselection)):
        #calculate the total portfolio budget by multiplying the budget of each project by the binary array obtained in projectselection    
        # print(projectselection[i])
        budgets[i] = np.sum(np.multiply(widened_bdgtperproject_matrix,projectselection[i]))
        #calculate the total portfolio npv by multiplying the npv of each project by the binary array obtained in projectselection
        PV_results[i] = np.sum(np.multiply(widened_PVperproject_matrix,projectselection[i]))
        #multiply dataframe 20r by the chosen portfolio to reflect the effect of the projects that are chosen
        pf_df20r = widened_df20r * projectselection[i]
        #sum the rows of the new dataframe to calculate the total cost of the portfolio
        pf_cost20r = pf_df20r.sum(axis=1)
        #extract the maximum of the resulting costs
        maxcost20r = max(pf_cost20r)
        # print("max cost:")
        # print(maxcost20r)
        #count how many results were higher than maxbdgt
        count = 0
        for j in range(pf_cost20r.__len__()):
            if pf_cost20r[j] > maxbdgt:
                count = count + 1
                # if the count is a multiple of 100, print the cost of the portfolio and the count
                # if count % 100 == 0:
                #    print("portfolio cost:", pf_cost20r[j])
                #    print("count: ", count)
        #array storing the portfolio risk not to exceed 10.800 Mio.€, as per-one risk units
        pf_conf2[i] = 1-count/iterations

    # create a dataframe with the results
    finalsol_df = pd.DataFrame({'Portfolio': projectselection, 'Portfolio PV': PV_results, 'Portfolio Budget': budgets, 'Portfolio confidence': pf_conf2})
    # order the dataframe by the portfolio PV, starting with the highest PV
    finalsol_df = finalsol_df.sort_values(by=['Portfolio PV'], ascending=False)
    print ("Final Solution: ", finalsol_df)

    PV_results = []
    budgets = []
    pf_cost20r = []
    #pf_conf2 = []

    #from the sorted dataframe, take the first row, which corresponds to the highest PV portfolio and extract the data needed for the following pictures
    finalsol_df = finalsol_df.iloc[0]
    best_stoch_pf = finalsol_df[0]
    PV_results_escalar = finalsol_df[1]
    PV_results.append(PV_results_escalar)
    #npv_results.append(finalsol_df[1])
    budgets_escalar = finalsol_df[2]
    budgets.append(budgets_escalar)
    #budgets.append(finalsol_df[2])
    # print ("Indexes of selected projects at deterministic portfolio: ", zipped_projection_indexes)
    print("portfolio_results: ", best_stoch_pf)
    print("PV_results: ", PV_results)
    print("budgets: ", budgets)
    
    
    # return(zipped_projection_indexes, budgets, PV_results, pf_conf2)
    return(best_stoch_pf, mcs_results2, widened_df20r)

    # from the projects at the selected portfolio, extract the costs and benefits of each project
    # and store them in a matrix, together with the project indexes

def simulatescenario(df10r, portfolio_projection, projectselection, iter):
    iterations = iter
    maxbdgt = 10800
    budgetting_confidence_policies = [0.75]


    #second simulation to get all cdfs for cost & benefits after optimization step
    mcs_results = simulate(portfolio_projection,iterations)

    # calculate the amount of projects in "portfolio_projection"
    projected_candidates = sum(portfolio_projection)

    # store the positions of the chosen projects in the portfolio_projection array, starting with 0 (as i+1 for if if starting with 1)
    zipped_projection_indexes = [i for i, x in enumerate(portfolio_projection) if x == 1]

    # mcs_results2[0] corresponds to the project costs and mcs_results2[1] to the project benefits (PV)
    x_perproj_matrix2 = pointestimate(mcs_results[0], mcs_results[1], budgetting_confidence_policies, projected_candidates)
    # print ("x_perproj_matrix2: ", x_perproj_matrix2)

    # we assume correlations at the cost side, not at the benefits side (conservative approach)
    # update x_perproj_matrix2 with the correlation effect registered inside df20r
    # print("x_perproj_matrix2: ", x_perproj_matrix2)
    # separate the budget and npv results from the x_perproj_matrix
    bdgtperproject_matrix = x_perproj_matrix2[0]
    PVperproject_matrix = x_perproj_matrix2[1]
    # print(type(bdgtperproject_matrix))
    # print(type(npvperproject_matrix))
    bdgtperproject_matrix = np.squeeze(bdgtperproject_matrix)
    PVperproject_matrix = np.squeeze(PVperproject_matrix)

    # remove all data that has zeroes from bdgtperproject_matrix and npvperproject_matrix
    # bdgtperproject_matrix = bdgtperproject_matrix[np.nonzero(bdgtperproject_matrix.flatten())]
    # npvperproject_matrix = npvperproject_matrix[np.nonzero(npvperproject_matrix.flatten())]

    # print("bdgtperproject_matrix: ", bdgtperproject_matrix)
    # print("npvperproject_matrix: ", npvperproject_matrix)
    # print("size of bdgtperproject_matrix", len(bdgtperproject_matrix))
    # print("size of npvperproject_matrix", len(npvperproject_matrix))
    # print("size of mcs_results2", len(mcs_results2))

    # print("mcs_results2 (input para correlacionar): ", mcs_results2)

    # for each of the options obtained in projectselection, calculate the total portfolio npv and the portfolio budget based on the information from x_perproj_matrix
    PV_results = [0] * len(projectselection) # as many as len(projectselection) because we have one npv per item in HoF
    budgets = [0] * len(projectselection)
    pf_conf2 = [0] * len(projectselection)
    widened_bdgtperproject_matrix = [0] * initcandidates # as many as initial amount of project candidates
    widened_PVperproject_matrix = [0] * initcandidates
    # initialize dataframe called widened_df20r as a copy of df10r
    widened_df20r = df10r.copy()
    # enlarge the dataframe to the size of iterations
    widened_df20r = widened_df20r.reindex(range(iterations))
    # fill the dataframe with zeroes
    widened_df20r.iloc[:, :] = 0

    df20r = correlatedMCS(mcs_results, iterations, projected_candidates, zipped_projection_indexes)
    # print("df20r: ", df20r)

    # pick in order the values from bdgtperproject_matrix and npvperproject_matrix and store them in widened_bdgtperproject_matrix and widened_npvperproject_matrix
    # The location of the values to be picked is available in zipped_projection_indexes
    j=0
    for i in range(initcandidates):
        if i in zipped_projection_indexes:
            widened_bdgtperproject_matrix [i] = round(bdgtperproject_matrix [j],3)
            widened_PVperproject_matrix [i] = round(PVperproject_matrix [j],3)
            j+=1
        else:
            pass
    # print("widened_bdgtperproject_matrix: ", widened_bdgtperproject_matrix)
    # print("widened_npvperproject_matrix: ", widened_npvperproject_matrix)

    # pick in order the values from df20r and store them in widened_df20r (to be used in the next step)
    i=0
    j=0
    k=0
    for i in range(initcandidates):
        if i in zipped_projection_indexes:
            for j in range(iterations):
                widened_df20r.loc[j, widened_df20r.columns[i]] = df20r.loc[j, df20r.columns[k]]
            k += 1
        else:
            pass

    # print("widened_df20r: ", widened_df20r)

    for i in range(len(projectselection)):
        #calculate the total portfolio budget by multiplying the budget of each project by the binary array obtained in projectselection    
        # print(projectselection[i])
        budgets[i] = np.sum(np.multiply(widened_bdgtperproject_matrix,projectselection[i]))
        #calculate the total portfolio npv by multiplying the npv of each project by the binary array obtained in projectselection
        PV_results[i] = np.sum(np.multiply(widened_PVperproject_matrix,projectselection[i]))
        #multiply dataframe 20r by the chosen portfolio to reflect the effect of the projects that are chosen
        pf_df20r = widened_df20r * projectselection[i]
        #sum the rows of the new dataframe to calculate the total cost of the portfolio
        pf_cost20r = pf_df20r.sum(axis=1)
        #extract the maximum of the resulting costs
        maxcost20r = max(pf_cost20r)
        # print("max cost:")
        # print(maxcost20r)
        #count how many results were higher than maxbdgt
        count = 0
        for j in range(pf_cost20r.__len__()):
            if pf_cost20r[j] > maxbdgt:
                count = count + 1
                # if the count is a multiple of 100, print the cost of the portfolio and the count
                # if count % 100 == 0:
                #    print("portfolio cost:", pf_cost20r[j])
                #    print("count: ", count)
        #array storing the portfolio risk not to exceed 10.800 Mio.€, as per-one risk units
        pf_conf2[i] = 1-count/iterations

    # create a dataframe with the results
    finalsol_df = pd.DataFrame({'Portfolio': projectselection, 'Portfolio PV': PV_results, 'Portfolio Budget': budgets, 'Portfolio confidence': pf_conf2})
    # order the dataframe by the portfolio PV, starting with the highest PV
    finalsol_df = finalsol_df.sort_values(by=['Portfolio PV'], ascending=False)
    # print ("Final Solution: ", finalsol_df)

    PV_results = []
    budgets = []
    pf_cost20r = []
    #pf_conf2 = []

    #from the sorted dataframe, take the first row, which corresponds to the highest PV portfolio and extract the data needed for the following pictures
    finalsol_df = finalsol_df.iloc[0]
    best_stoch_pf = finalsol_df[0]
    PV_results_escalar = finalsol_df[1]
    PV_results.append(PV_results_escalar)
    #npv_results.append(finalsol_df[1])
    budgets_escalar = finalsol_df[2]
    budgets.append(budgets_escalar)
    #budgets.append(finalsol_df[2])
    # print ("Indexes of selected projects at deterministic portfolio: ", zipped_projection_indexes)
    # print("portfolio_results: ", portfolio_results)
    # print("npv_results: ", npv_results)
    # print("budgets: ", budgets)
    
    
    return(mcs_results, widened_df20r)

    # from the projects at the selected portfolio, extract the costs and benefits of each project
    # and store them in a matrix, together with the project indexes