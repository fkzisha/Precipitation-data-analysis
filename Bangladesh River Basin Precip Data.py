import pandas as pd
import numpy as np
import os

# Define the directory where your files are located
directory = "C:/5. Thesis/Prec Data/Prec_long term/PP"

precip_file_names = [f for f in os.listdir(directory) if f[0].isdigit() and f.endswith(".csv")]

# Create a list of all date-time file names (assuming they follow a specific pattern)
datetime_file_names = [f for f in os.listdir(directory) if f.startswith("time") and f.endswith(".csv")]

# Loop through each pair of precipitation and date-time files
for precip_file_name, datetime_file_name in zip(precip_file_names, datetime_file_names):
    # Read the target location data
    target_locations = pd.read_csv(os.path.join(directory, "cord_BD_rb.csv"))

    # Read the precipitation data and set the first column as the index
    daily_precip_data = pd.read_csv(os.path.join(directory, precip_file_name), header=0, index_col=0)

    # Extract the specific indexes from the target_locations DataFrame
    selected_indexes = target_locations['Order_']

    # Select the desired columns from the precipitation data using the selected indexes
    selected_precipitation_data = daily_precip_data.iloc[:, selected_indexes]

    # Define the path to the new CSV file for selected precipitation data
    new_precip_file_path = os.path.join(directory, f"Sorted Data_RB/{precip_file_name}")

    # Save the selected precipitation data to the new file
    selected_precipitation_data.to_csv(new_precip_file_path)

    # Read the date and time data from the separate file
    date_time_data = pd.read_csv(os.path.join(directory, datetime_file_name))

    # Add a new column 'Identifier' to both DataFrames with the same values
    selected_precipitation_data.loc[:, 'Identifier'] = np.arange(len(selected_precipitation_data))
    date_time_data['Identifier'] = np.arange(len(date_time_data))

    # Combine the two DataFrames using 'Identifier' as the common column
    combined_data = pd.merge(date_time_data, selected_precipitation_data, on='Identifier')

    # Drop the 'Identifier' column if you no longer need it
    combined_data = combined_data.drop(columns=['Identifier'])

    # Define the column names
    column_names = ['Year', 'Month', 'Date', 'Hour'] + [f'Lat_Long_{i}' for i in range(1, len(combined_data.columns) - 3)]

    # Assign the column names to the DataFrame
    combined_data.columns = column_names

    # Define the path to the new CSV file for combined data
    new_combined_file_path = os.path.join(directory, f"Combined Data_RB/combined_{precip_file_name}")

    # Save the combined data to the new file
    combined_data.to_csv(new_combined_file_path, index=False)

print("All files processed successfully.")

import os
import pandas as pd
import numpy as np
import arcpy

# Define the path to your precipitation data file
directory = "C:/5. Thesis/River Basin/Combined Data_RB"

# List all CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Loop through each CSV file
for csv_file in csv_files:
    # Define the path to the current CSV file
    csv_file_path = os.path.join(directory, csv_file)

    # Read the CSV data into a DataFrame
    precipitation_df = pd.read_csv(csv_file_path)

    # Replace negative or exponential values with "NaN" strings
    precipitation_df[precipitation_df < 0] = np.nan
    precipitation_df[precipitation_df > 1e6] = np.nan

    # Save the modified DataFrame back to the same CSV file
    precipitation_df.to_csv(csv_file_path, index=False, na_rep="NaN")

# Print a message to indicate that all files have been processed
print("All CSV files processed successfully.")

import pandas as pd
import os

# Directory where your CSV files are located
data_dir = 'C:/5. Thesis/River Basin/Combined Data_RB'

# Initialize an empty DataFrame for combined data
combined_data = pd.DataFrame(columns=['Year', 'Precipitation'])

# List all CSV files in the directory
csv_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]

# Loop through each CSV file and aggregate precipitation data
for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)
    df = pd.read_csv(file_path)  # Read the CSV file
    # Assuming "Lat-Long" columns are from the 5th column onward (adjust as needed)
    precipitation_data = df.iloc[:, 4:].sum(axis=1)
    year_data = df['Year']
    
    # Combine data with the existing DataFrame
    combined_data = pd.concat([combined_data, pd.DataFrame({'Year': year_data, 'Precipitation': precipitation_data})])

    # Define the path to the new Excel file
output_file = "C:/5. Thesis/River Basin/Analysis/YW_Prec.xlsx"

# Save the combined data to a new CSV file
combined_data.to_excel(output_file, index=False)

import pandas as pd

# Read your original data file (replace 'your_data.csv' with your actual file path)
data = pd.read_excel("C:/5. Thesis/River Basin/Analysis/YW_Prec.xlsx")

# Group the data by year and calculate the mean of the precipitation data for each year
grouped_data = data.groupby('Year')['Precipitation'].agg(['sum', 'count']).reset_index()
grouped_data['Precipitation'] = grouped_data['sum'] / grouped_data['count']/985
grouped_data.drop(['sum', 'count'], axis=1, inplace=True)

# Define the path to the new Excel file
output_file = "C:/5. Thesis/River Basin/Analysis/YW_Prec_AggAvg.xlsx"

# Save the modified data to a new excel file
grouped_data.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, linregress

# Load your data (replace 'your_data.csv' with the actual file path)
file_path = 'C:/5. Thesis/River Basin/Analysis/YW_Prec_AggAvg.xlsx'
data = pd.read_excel(file_path)

# Extract year and precipitation columns
years = data['Year']
precipitation = data['Precipitation']

# Calculate Sen's Slope
def sen_slope(y):
    n = len(y)
    slopes = [np.nan] * n
    for i in range(n):
        for j in range(i + 1, n):
            slopes[j - i - 1] = (y[j] - y[i]) / (j - i)
    return np.nanmedian(slopes)

sen = sen_slope(precipitation)

# Perform Mann-Kendall trend test
tau, p_value = kendalltau(years, precipitation)

# Output results
print(f"Sen's Slope: {sen:.4f}")
print(f"Mann-Kendall Tau: {tau:.4f}")
print(f"P-Value: {p_value:.4f}")

# Additional: Linear Regression for trend line
slope, intercept, _, _, _ = linregress(years, precipitation)
print(f"Linear Regression Slope: {slope:.4f}")
print(f"Linear Regression Intercept: {intercept:.4f}")

alpha = 0.05  # significance level

if p_value < alpha:
    print("The Mann-Kendall test result is significant. Reject the null hypothesis.")
else:
    print("The Mann-Kendall test result is not significant. Fail to reject the null hypothesis.")

import pandas as pd
import numpy as np
import os

# Define the directory where your 34 CSV files are located
directory = "C:/5. Thesis/River Basin/Combined Data_RB"

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Initialize a DataFrame to store the aggregated results
data = pd.DataFrame()

# Loop through each CSV file
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(directory, csv_file))

    # get the precipitation for each column except Month, Date, and Hour
    sum_precip = df.drop(columns=["Month", "Date", "Hour"])

    # Extract the Year from the first row (assuming it's the same for all rows)
    year = df.loc[0, "Year"]

    # Add the Year as a separate column
    sum_precip["Year"] = year

    # Add the result to the DataFrame
    data = data.append(sum_precip, ignore_index=True)

# Define the path to save the data
output_file = "C:/5. Thesis/River Basin/Analysis/Combined_RB.xlsx"

# Save the data to a exce file
data.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
import os

# Define the directory where your 34 CSV files are located
directory = "C:/5. Thesis/River Basin/Combined Data_RB"

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Initialize a DataFrame to store the aggregated results
aggregated_data = pd.DataFrame()

# Loop through each CSV file
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(directory, csv_file))

    # Sum the precipitation for each column except Month, Date, and Hour
    sum_precip = df.drop(columns=["Month", "Date", "Hour"]).sum()

    # Extract the Year from the first row (assuming it's the same for all rows)
    year = df.loc[0, "Year"]

     # Calculate the count of rows for each year
    count_per_year = df.groupby('Year').size().get(year, 0)

    # Calculate the average precipitation by dividing the sum by the count
    avg_precip = sum_precip / count_per_year

    # Add the Year as a separate column
    avg_precip["Year"] = year

    # Add the result to the aggregated DataFrame
    aggregated_data = aggregated_data.append(avg_precip, ignore_index=True)

# Define the path to save the aggregated data
output_file = "C:/5. Thesis/River Basin/Analysis/Presample_Percentiles.xlsx"

# Save the aggregated data to a CSV file
aggregated_data.to_excel(output_file, index=False)

import pandas as pd

# Load your data into a DataFrame (assuming your data is in an Excel file)
data_file = "C:/5. Thesis/River Basin/Analysis/Presample_Percentiles.xlsx"
df = pd.read_excel(data_file) 

# Create a new DataFrame to store the combined data
new_df = pd.DataFrame()

# Combine all the columns into a single column
precipitation_values = df.drop('Year', axis=1).values.ravel()

# Add the combined precipitation values to the new DataFrame
new_df['Precipitation'] = precipitation_values

# Add the 'Year' column to the new DataFrame
new_df['Year'] = df['Year'].repeat(len(df.columns) - 1).reset_index(drop=True)

# Define the path to the new Excel file
output_file = "C:/5. Thesis/River Basin/Analysis/Sample_Percentiles.xlsx"

# Save the aggregated data to the new Excel file
new_df.to_excel(output_file, index=False)

import numpy as np
import pandas as pd

# Read your data from the CSV file
data_file = "C:/5. Thesis/River Basin/Analysis/Sample_Percentiles.xlsx"
df = pd.read_excel(data_file)

# Remove the 'Year' column from the DataFrame
data = df.drop(columns=['Year'])

# Flatten the data into a 1D array
data_flat = data.values.flatten()

# Calculate percentiles
percentile_75 = np.percentile(data_flat, 75)
percentile_90 = np.percentile(data_flat, 90)
percentile_95 = np.percentile(data_flat, 95)

# Print the results
print(f"75th Percentile: {percentile_75}")
print(f"90th Percentile: {percentile_90}")
print(f"95th Percentile: {percentile_95}")

import pandas as pd

# Load the aggregated data file
data_file = "C:/5. Thesis/River Basin/Analysis/Sample_Percentiles.xlsx"
df = pd.read_excel(data_file)

# Create a sample DataFrame with values greater than or equal to 9.514186123227114
sample_df = df[df['Precipitation'] >= 9.514186123227114]

# Define the path to save the sample
sample_file = "C:/5. Thesis/River Basin/Analysis/90th Percentiles.xlsx"

# Save the sample DataFrame to a new CSV file
sample_df.to_excel(sample_file, index=False)

import pandas as pd

# Read your original data file 
data = pd.read_excel("C:/5. Thesis/River Basin/Analysis/90th Percentiles.xlsx")

# Group the data by year and calculate the mean of the precipitation data for each year
grouped_data = data.groupby('Year')['Precipitation'].agg(['sum', 'count']).reset_index()
grouped_data['Precipitation'] = grouped_data['sum'] / grouped_data['count']
grouped_data.drop(['sum', 'count'], axis=1, inplace=True)

# Define the path to the new Excel file
output_file = "C:/5. Thesis/River Basin/Analysis/90th_Per_Yearwise.xlsx"

# Save the modified data to a new excel file
grouped_data.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, linregress

# Load your data (replace 'your_data.csv' with the actual file path)
file_path = 'C:/5. Thesis/River Basin/Analysis/90th_Per_Yearwise.xlsx'
data = pd.read_excel(file_path)

# Extract year and precipitation columns
years = data['Year']
precipitation = data['Precipitation']

# Calculate Sen's Slope
def sen_slope(y):
    n = len(y)
    slopes = [np.nan] * n
    for i in range(n):
        for j in range(i + 1, n):
            slopes[j - i - 1] = (y[j] - y[i]) / (j - i)
    return np.nanmedian(slopes)

sen = sen_slope(precipitation)

# Perform Mann-Kendall trend test
tau, p_value = kendalltau(years, precipitation)

# Output results
print(f"Sen's Slope: {sen:.4f}")
print(f"Mann-Kendall Tau: {tau:.4f}")
print(f"P-Value: {p_value:.4f}")

# Additional: Linear Regression for trend line
slope, intercept, _, _, _ = linregress(years, precipitation)
print(f"Linear Regression Slope: {slope:.4f}")
print(f"Linear Regression Intercept: {intercept:.4f}")

alpha = 0.05  # significance level

if p_value < alpha:
    print("The Mann-Kendall test result is significant. Reject the null hypothesis.")
else:
    print("The Mann-Kendall test result is not significant. Fail to reject the null hypothesis.")

import pandas as pd

# Load the aggregated data file
data_file = "C:/5. Thesis/River Basin/Analysis/Sample_Percentiles.xlsx"
df = pd.read_excel(data_file)

# Create a sample DataFrame with values greater than or equal to 11.004082718209146
sample_df = df[df['Precipitation'] >= 11.004082718209146]

# Define the path to save the sample
sample_file = "C:/5. Thesis/River Basin/Analysis/95th Percentiles.xlsx"

# Save the sample DataFrame to a new CSV file
sample_df.to_excel(sample_file, index=False)

import pandas as pd

# Read your original data file 
data = pd.read_excel("C:/5. Thesis/River Basin/Analysis/95th Percentiles.xlsx")

# Group the data by year and calculate the mean of the precipitation data for each year
grouped_data = data.groupby('Year')['Precipitation'].agg(['sum', 'count']).reset_index()
grouped_data['Precipitation'] = grouped_data['sum'] / grouped_data['count']
grouped_data.drop(['sum', 'count'], axis=1, inplace=True)

# Define the path to the new Excel file
output_file = "C:/5. Thesis/River Basin/Analysis/95th_Per_Yearwise.xlsx"

# Save the modified data to a new excel file
grouped_data.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, linregress

# Load your data (replace 'your_data.csv' with the actual file path)
file_path = 'C:/5. Thesis/River Basin/Analysis/95th_Per_Yearwise.xlsx'
data = pd.read_excel(file_path)

# Extract year and precipitation columns
years = data['Year']
precipitation = data['Precipitation']

# Calculate Sen's Slope
def sen_slope(y):
    n = len(y)
    slopes = [np.nan] * n
    for i in range(n):
        for j in range(i + 1, n):
            slopes[j - i - 1] = (y[j] - y[i]) / (j - i)
    return np.nanmedian(slopes)

sen = sen_slope(precipitation)

# Perform Mann-Kendall trend test
tau, p_value = kendalltau(years, precipitation)

# Output results
print(f"Sen's Slope: {sen:.4f}")
print(f"Mann-Kendall Tau: {tau:.4f}")
print(f"P-Value: {p_value:.4f}")

# Additional: Linear Regression for trend line
slope, intercept, _, _, _ = linregress(years, precipitation)
print(f"Linear Regression Slope: {slope:.4f}")
print(f"Linear Regression Intercept: {intercept:.4f}")

alpha = 0.05  # significance level

if p_value < alpha:
    print("The Mann-Kendall test result is significant. Reject the null hypothesis.")
else:
    print("The Mann-Kendall test result is not significant. Fail to reject the null hypothesis.")

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, linregress

# Load your data (replace 'your_data.csv' with the actual file path)
file_path = 'C:/5. Thesis/River Basin/Analysis/95th_Per_Yearwise.xlsx'
data = pd.read_excel(file_path)

# Filter data for the years 2000-2022
filtered_data = data[(data['Year'] >= 2000) & (data['Year'] <= 2022)]

# Extract year and precipitation columns
years = filtered_data['Year']
precipitation = filtered_data['Precipitation']

def sen_slope(y):
    n = len(y)
    slopes = [np.nan] * n
    for i in range(n):
        for j in range(i + 1, n):
            slopes[j - i - 1] = (y.iloc[j] - y.iloc[i]) / (j - i)
    return np.nanmedian(slopes)

sen = sen_slope(precipitation)

# Perform Mann-Kendall trend test
tau, p_value = kendalltau(years, precipitation)

# Output results
print(f"Sen's Slope: {sen:.4f}")
print(f"Mann-Kendall Tau: {tau:.4f}")
print(f"P-Value: {p_value:.4f}")

# Additional: Linear Regression for trend line
slope, intercept, _, _, _ = linregress(years, precipitation)
print(f"Linear Regression Slope: {slope:.4f}")
print(f"Linear Regression Intercept: {intercept:.4f}")

alpha = 0.05  # significance level

if p_value < alpha:
    print("The Mann-Kendall test result is significant. Reject the null hypothesis.")
else:
    print("The Mann-Kendall test result is not significant. Fail to reject the null hypothesis.")

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, linregress

# Load your data (replace 'your_data.csv' with the actual file path)
file_path = 'C:/5. Thesis/River Basin/Analysis/YW_Prec_AggAvg.xlsx'
data = pd.read_excel(file_path)

# Filter data for the years 2000-2022
filtered_data = data[(data['Year'] >= 2000) & (data['Year'] <= 2022)]

# Extract year and precipitation columns
years = filtered_data['Year']
precipitation = filtered_data['Precipitation']

def sen_slope(y):
    n = len(y)
    slopes = [np.nan] * n
    for i in range(n):
        for j in range(i + 1, n):
            slopes[j - i - 1] = (y.iloc[j] - y.iloc[i]) / (j - i)
    return np.nanmedian(slopes)

sen = sen_slope(precipitation)

# Perform Mann-Kendall trend test
tau, p_value = kendalltau(years, precipitation)

# Output results
print(f"Sen's Slope: {sen:.4f}")
print(f"Mann-Kendall Tau: {tau:.4f}")
print(f"P-Value: {p_value:.4f}")

# Additional: Linear Regression for trend line
slope, intercept, _, _, _ = linregress(years, precipitation)
print(f"Linear Regression Slope: {slope:.4f}")
print(f"Linear Regression Intercept: {intercept:.4f}")

alpha = 0.05  # significance level

if p_value < alpha:
    print("The Mann-Kendall test result is significant. Reject the null hypothesis.")
else:
    print("The Mann-Kendall test result is not significant. Fail to reject the null hypothesis.")

import pandas as pd

# Read your original data file (replace 'your_data.csv' with your actual file path)
data = pd.read_excel("C:/5. Thesis/River Basin/Analysis/YW_Prec.xlsx")

# Group the data by year and calculate the mean of the precipitation data for each year
grouped_data = data.groupby('Year')['Precipitation'].agg(['sum', 'count']).reset_index()
grouped_data['Precipitation'] = grouped_data['sum'] / grouped_data['count']
grouped_data.drop(['sum', 'count'], axis=1, inplace=True)

# Define the path to the new Excel file
output_file = "C:/5. Thesis/River Basin/Analysis/Accum_RB.xlsx"

# Save the modified data to a new excel file
grouped_data.to_excel(output_file, index=False)
