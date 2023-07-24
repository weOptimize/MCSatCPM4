import os
import random
import shutil
import pandas as pd
import numpy as np
import math
import pyexcel as pe
import pyexcel_ods3

source_dir = r"C:\REPO_CLONES\MCSatCPM4\RND_Schedules\CSV"
# choose the id of the folder and store as variable
id_folder = 2
# create the destination folder and name it "New_Files" + id_folder
dest_dir = r"C:\REPO_CLONES\MCSatCPM4\RND_Schedules\CSV\New_Files" + str(id_folder)

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)


# create a new dataframe and store the number of the file and the sum of the values in the "MODE" column
proj_overview = pd.DataFrame(columns=["File", "Total_days_Sum"])
for filename in os.listdir(dest_dir):
    if filename.startswith("data_wb") and filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(dest_dir, filename))
        proj_overview = pd.concat([proj_overview, pd.DataFrame({"File": [filename], "Total_days_Sum": [df["MODE"].sum()]})], ignore_index=True)
        print (proj_overview)




# Load the expected cash flows file located at sourcedir
cf_filename = "expected_cash_flows.txt"
with open(os.path.join(source_dir, cf_filename), "r") as f:
    cash_flows = f.readlines()


# Normalize the Total_days_Sum column to have values between 0 and 1
proj_overview['Normalized_days'] = proj_overview['Total_days_Sum'] / proj_overview['Total_days_Sum'].max()

# Generate random cash flows for 4 years with a weak correlation to Total_days_Sum
cash_flows = []
for index, row in proj_overview.iterrows():
    base_cash_flow = round(row['Normalized_days'],2) * 1500  # Arbitrary scaling factor
    year_1 = int(base_cash_flow * (1 + round(np.random.uniform(0, 0.12),2)))
    year_2 = int(base_cash_flow * (1 + round(np.random.uniform(0.1, 0.23),2)))
    year_3 = int(base_cash_flow * (1 + round(np.random.uniform(0.2, 0.35),2)))
    year_4 = int(base_cash_flow * (1 + round(np.random.uniform(0.3, 0.48),2)))  
    cash_flows.append([year_1, year_2, year_3, year_4])

# Create a DataFrame with the generated cash flows and save it as a txt file
cash_flows_df = pd.DataFrame(cash_flows, columns=['Year_1', 'Year_2', 'Year_3', 'Year_4'])
print(cash_flows_df)
#store the cash flows in a txt file located in the same folder as expected_cash_flows.txt use spaces as separator
cash_flows_df.to_csv(os.path.join(source_dir, "cash_flows.txt"), sep=" ", index=False)
