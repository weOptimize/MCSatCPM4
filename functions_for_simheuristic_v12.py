#This file includes the function that returns the survival value for a given budgetting confidence policy.
#It is called from the main file
import math
import random
import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from operator import itemgetter
from pandas_ods_reader import read_ods
from copulas.multivariate import GaussianMultivariate
from scipy.stats import rv_continuous, rv_histogram, norm, uniform, multivariate_normal, beta
from fitter import Fitter, get_common_distributions, get_distributions

#import created scripts:
from task_rnd_triang_with_interrupts_stdev_new_R2 import *

#I define the number of candidates to be considered
nrcandidates = 10
nr_confidence_policies = 1
mcs_costs = []
mcs_NPV = []
maxbdgt = 3800
#initialize matrices to store bdgt and npv
bdgtperproject_matrix = np.zeros((nrcandidates, nr_confidence_policies))
npvperproject_matrix = np.zeros((nrcandidates, nr_confidence_policies))


#defining the function that calculates the total budget of a portfolio of projects
def portfolio_totalbudget(portfolio,bdgtperproject):
    totalbudget_portfolio = 0
    #totalbudget_npv = 0
    for i in range(nrcandidates):
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
def expected_value_extractor(sim_npv, iterations):
    #calculate the cumulative sum of the values of the histogram
	valuesplus, base = np.histogram(sim_npv, bins=iterations) #it returns as many values as specified in bins valuesplus are frequencies, base the x-axis limits for the bins 
	cumulativeplus = np.cumsum(valuesplus)
	survivalvalues = 100*(len(sim_npv)-cumulativeplus)/len(sim_npv)
	#return index of item from survivalvalues that is closest to "1-budgetting_confidence_policy" typ.20%
	index = (np.abs(survivalvalues-100*(1-.5))).argmin()
	#return value at base (which is indeed the durations that correspond to survival level) that matches the index
	budgetedduration = np.round(base[index],2)
	return budgetedduration



def simulate(arrayforsim, iterat):
    #initialize the arrays that will store the results of the MonteCarlo Simulation
    mcs_costs = []
    mcs_NPV = []
    for i in range(len(arrayforsim)):        
        #if the value i is 1, then the simulation is performed
        if arrayforsim[i] == 1:
            #open ten different ODS files and store the results in a list after computing the CPM and MCS
            filename = "RND_Schedules/data_wb" + str(i+1) + ".ods"
            #print(filename)
            mydata = read_ods(filename, "Sheet1")
            #open ten different ODS files and store the results in a list after computing the CPM and MCS
            filename = "RND_Schedules/riskreg_" + str(i+1) + ".ods"
            #print(filename)
            myriskreg = read_ods(filename, "riskreg")
            #compute MonteCarlo Simulation and store the results in an array called "sim1_costs"
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

            # compute MonteCarlo Simulation and store the results in an array called "sim1_NPV"
            #print(cashflows)
            sim_NPV = MCS_NPV(cashflows, iterat)
            #print(sim_NPV)
            
            #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the cost at each iteration
            mcs_costs.append(sim_costs)
            mcs_NPV.append(sim_NPV)
            #store each of the results from the MCS in an array where the columns correspond to the projects and the rows correspond to the NPV at each iteration
            #mcs_npvs1.append(sim1_NPV)
            #compute the median of the NPV results
            median_npv = expected_value_extractor(sim_NPV, iterat)
        else:
            #if the value i is 0, then the simulation is not performed and the appended results an array full of zeros
            mcs_NPV.append([0.0])   
            mcs_costs.append(np.zeros(iterat))
         
            

    #print ("mcs_costs", mcs_costs)
    #print ("mcs_NPV", mcs_NPV)
    return(mcs_costs, mcs_NPV)

# compute the median of the NPV results
def pointestimate(mcs_costs, mcs_NPV, budgetting_confidence_policies):
    for i in range(nrcandidates):
        median_npv = round(expected_value_extractor(mcs_NPV[i], len(mcs_NPV[i])),0)
        for j in range(len(budgetting_confidence_policies)):
            budgetting_confidence_policy = budgetting_confidence_policies[j]
            #extract the survival value from the array sim_duration that corresponds to the budgetting confidence policy
            survival_value = survival_value_extractor(mcs_costs[i], budgetting_confidence_policy, len(mcs_costs[i]))
            #store the first survival value in an array where the columns correspond to the budgetting confidence policies and the rows correspond to the projects
            bdgtperproject_matrix[i][j]=survival_value
            npvperproject_matrix[i][j]=median_npv-survival_value
    return(bdgtperproject_matrix, npvperproject_matrix)

# modify MCS results to reflect the correlation matrix
def correlatedMCS(mcs_results, iterat):
    #check the parameters of beta distribution for each of the mcs_results
    betaparams = []
    for i in range(nrcandidates):
        f = Fitter(mcs_results[0][i], distributions=['beta'])
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
    df0 = pd.DataFrame(data=mcs_results[0]).T
    df0.rename(columns={0:"P01", 1:"P02", 2:"P03", 3:"P04", 4:"P05", 5:"P06", 6:"P07", 7:"P08", 8:"P09", 9:"P10"}, inplace=True)
    correlation_matrix0 = df0.corr()

    # *********Correlation matrix with random values between 0 and 1, but positive semidefinite***************
    # Set the seed value for the random number generator  
    seed_value = 1005  
    np.random.seed(seed_value)
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

    #initialize dataframe df10r with size nrcandidates x iterations
    df10r = pd.DataFrame(np.zeros((iterat, nrcandidates)))
    # step 1: draw random variates from a multivariate normal distribution 
    # with the targeted correlation structure
    r0 = [0] * cm10r.shape[0]                       # create vector r with as many zeros as correlation matrix has variables (row or columns)
    mv_norm = multivariate_normal(mean=r0, cov=cm10r)    # means = vector of zeros; cov = targeted corr matrix
    rand_Nmv = mv_norm.rvs(iterat)                               # draw N random variates
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
    rand_P1 = d_P1.rvs(iterat)
    rand_P2 = d_P2.rvs(iterat)
    rand_P3 = d_P3.rvs(iterat)
    rand_P4 = d_P4.rvs(iterat)
    rand_P5 = d_P5.rvs(iterat)
    rand_P6 = d_P6.rvs(iterat)
    rand_P7 = d_P7.rvs(iterat)
    rand_P8 = d_P8.rvs(iterat)
    rand_P9 = d_P9.rvs(iterat)
    rand_P10 = d_P10.rvs(iterat)
    # initial correlation structure before applying a copula
    c_before = np.corrcoef([rand_P1, rand_P2, rand_P3, rand_P4, rand_P5, rand_P6, rand_P7, rand_P8, rand_P9, rand_P10])
    # step 4: draw N random variates for each of the three marginal distributions
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
    correlation_matrix1 = df10r.corr()
    return df10r