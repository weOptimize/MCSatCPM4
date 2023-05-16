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
            mcs_costs.append([])
            mcs_NPV.append([])
    return(mcs_costs, mcs_NPV)

#compute the median of the NPV results
def pointestimate(mcs_costs, mcs_NPV, budgetting_confidence_policies):
    for i in range(nrcandidates):
        median_npv = expected_value_extractor(mcs_NPV[i], len(mcs_NPV[i]))
        for j in range(len(budgetting_confidence_policies)):
            budgetting_confidence_policy = budgetting_confidence_policies[j]
            #extract the survival value from the array sim_duration that corresponds to the budgetting confidence policy
            survival_value = survival_value_extractor(mcs_costs[i], budgetting_confidence_policy, len(mcs_costs[i]))
            #store the first survival value in an array where the columns correspond to the budgetting confidence policies and the rows correspond to the projects
            bdgtperproject_matrix[i][j]=survival_value
            npvperproject_matrix[i][j]=median_npv-survival_value
    return(bdgtperproject_matrix, npvperproject_matrix)