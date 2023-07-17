import numpy as np
import random
from scipy.optimize import linprog

# Load input data
baseline_costs = ...  # Load baseline costs (1x20)
daily_cost = 500
schedules = ...  # Load schedules (list of 20 lists containing tuples of (shortest, longest, mode))
risks = ...  # Load risks (list of 20 lists containing 10 tuples of (probability, worst_case, best_case, mode))
correlations = ...  # Load correlation matrix (20x20)
total_cost_limit = ...  # Set the total cost limit
cash_flows = ...  # Load cash flows (20x4)
WACC = ...  # Load the Weighted Average Cost of Capital

# Number of scenarios
num_scenarios = 1000

# Calculate NPV for each project
def calculate_npv(cash_flows, WACC):
    discount_factors = [(1 + WACC) ** -(i + 1) for i in range(cash_flows.shape[1])]
    npvs = cash_flows * discount_factors
    return np.sum(npvs, axis=1)

npvs = calculate_npv(cash_flows, WACC)

# Define the optimization function
def optimize_portfolio(costs, npvs, total_cost_limit):
    num_projects = len(npvs)
    c = -npvs
    A = [costs]
    b = [total_cost_limit]

    bounds = [(0, 1) for _ in range(num_projects)]

    res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs', integrality=np.ones(num_projects), dualize=True)

    return res.x

# Main loop to generate scenarios and optimize the portfolio
best_portfolio = None
best_portfolio_npv = -np.inf

for _ in range(num_scenarios):
    # Generate random scenarios for risks and schedule durations
    scenario_risks = [random.choices(risk, k=1)[0] for risk in risks]
    scenario_durations = [random.triangular(shortest, longest, mode) for shortest, longest, mode in schedules]

    # Calculate additional costs for each project
    additional_risk_costs = [risk[2] * risk[0] for risk in scenario_risks]
    additional_schedule_costs = [duration * daily_cost for duration in scenario_durations]
    total_additional_costs = np.array(additional_risk_costs) + np.array(additional_schedule_costs)

    # Calculate the total costs for each project
    total_costs = baseline_costs + total_additional_costs

    # Optimize the portfolio for this scenario
    portfolio = optimize_portfolio(total_costs, npvs, total_cost_limit)

    # Calculate the NPV of the optimized portfolio
    portfolio_npv = np.sum(portfolio * npvs)

    # Update the best portfolio if necessary
    if portfolio_npv > best_portfolio_npv:
        best_portfolio_npv = portfolio_npv
        best_portfolio = portfolio

print("Best portfolio:", best_portfolio)
print("Best portfolio NPV:", best_portfolio_npv)

# generate file output with portolio and npv arrays, store with name "portfolio.npy"
np.save("portfolio.npy", best_portfolio)
np.save("npv.npy", best_portfolio_npv)
