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
    target_locations = pd.read_csv(os.path.join(directory, "cord.csv"))

    # Read the precipitation data and set the first column as the index
    daily_precip_data = pd.read_csv(os.path.join(directory, precip_file_name), header=0, index_col=0)

    # Extract the specific indexes from the target_locations DataFrame
    selected_indexes = target_locations['Order_']

    # Select the desired columns from the precipitation data using the selected indexes
    selected_precipitation_data = daily_precip_data.iloc[:, selected_indexes]

    # Define the path to the new CSV file for selected precipitation data
    new_precip_file_path = os.path.join(directory, f"Sorted Data/{precip_file_name}")

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
    new_combined_file_path = os.path.join(directory, f"Combined data/combined_{precip_file_name}")

    # Save the combined data to the new file
    combined_data.to_csv(new_combined_file_path, index=False)

print("All files processed successfully.")

import os
import pandas as pd
import numpy as np
import arcpy

# Define the path to your precipitation data file
directory = "C:/5. Thesis/Prec Data/Prec_long term/PP/Combined data"

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
import numpy as np
import os

# Define the directory where your 34 CSV files are located
directory = "C:/5. Thesis/Prec Data/Prec_long term/Combined data"

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

    # Add the Year as a separate column
    sum_precip["Year"] = year

    # Add the result to the aggregated DataFrame
    aggregated_data = aggregated_data.append(sum_precip, ignore_index=True)

# Define the path to save the aggregated data
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Aggregated_LTP.xlsx"

# Save the aggregated data to a CSV file
aggregated_data.to_excel(output_file, index=False)

import pandas as pd

# Load your data into a DataFrame
data_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Aggregated_LTP.xlsx"
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
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/timeseries_presample_LTP.xlsx"

# Save the aggregated data to the new Excel file
new_df.to_excel(output_file, index=False)

import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

# Load the data from the file
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/timeseries_presample_LTP.xlsx")

# Ensure that the 'Year' column is of integer type
df['Year'] = df['Year'].astype(int)

# Filter the data for the years 1950 to 1987
start_year = 1950
end_year = 1987
filtered_data = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Sort the data by year
filtered_data = filtered_data.sort_values(by='Year')

plt.figure(figsize=(12, 6))
plt.bar(filtered_data['Year'], filtered_data['Precipitation'], label="Precipitation")
plt.xlabel("Year")
plt.ylabel("Precipitation")
plt.title("Precipitation Bar Chart (1950-1987)")
plt.legend()

# Calculate the trendline
x = np.array(filtered_data['Year'])
y = np.array(filtered_data['Precipitation'])
b, m = polyfit(x, y, 1)

# Add the trendline to the plot
plt.plot(x, b + m * x, color='red', label=f'Trendline (y = {m:.2f}x + {b:.2f})')
plt.legend()

plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyfit

# Load the data from the file
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/timeseries_presample_LTP.xlsx")

# Ensure that the 'Year' column is of integer type
df['Year'] = df['Year'].astype(int)

# Filter the data for the years 1950 to 1987
start_year = 1988
end_year = 2022
filtered_data = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Sort the data by year
filtered_data = filtered_data.sort_values(by='Year')

plt.figure(figsize=(12, 6))
plt.bar(filtered_data['Year'], filtered_data['Precipitation'], label="Precipitation")
plt.xlabel("Year")
plt.ylabel("Precipitation")
plt.title("Precipitation Bar Chart (1950-1987)")
plt.legend()

# Calculate the trendline
x = np.array(filtered_data['Year'])
y = np.array(filtered_data['Precipitation'])
b, m = polyfit(x, y, 1)

# Add the trendline to the plot
plt.plot(x, b + m * x, color='red', label=f'Trendline (y = {m:.2f}x + {b:.2f})')
plt.legend()

plt.grid(True)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the data from the file
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/timeseries_presample_LTP.xlsx")

# Filter data for the desired time range (1950-1987)
df_filtered = df[(df['Year'] >= 1950) & (df['Year'] <= 1987)]

plt.figure(figsize=(10, 6))

# Scatter plot of the data
plt.scatter(df_filtered['Year'], df_filtered['Precipitation'], label="Data")

# Linear regression line
regression_line = intercept + slope * df_filtered['Year']
plt.plot(df_filtered['Year'], regression_line, 'r', label="Linear Regression Line")

# Calculate R-squared value
r_squared = r_value**2

# Add R-squared value to the chart title
plt.title(f'Time Series of Precipitation (1950-1987)\nR-squared = {r_squared:.4f}')
plt.xlabel('Year')
plt.ylabel('Precipitation')
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load your data into a Pandas DataFrame (replace 'your_data.csv' with your actual file)
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/timeseries_presample_LTP.xlsx")

# Filter data for the desired time range (1950-1987)
df_filtered = df[(df['Year'] >= 1950) & (df['Year'] <= 1987)]

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(df_filtered['Year'], df_filtered['Precipitation'])

# Calculate R-squared value
r_squared = r_value**2

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(df_filtered['Year'], df_filtered['Precipitation'], label="Precipitation")

# Add the R-squared value to the chart title
plt.title(f'Year-wise Precipitation (1950-1987)\nR-squared = {r_squared:.4f}')
plt.xlabel('Year')
plt.ylabel('Precipitation')
plt.legend()
plt.grid(axis='y')
plt.show()

import pandas as pd
import numpy as np

# Read your time series data into a DataFrame
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/timeseries_presample_LTP.xlsx")

# Filter the data for the desired year range (e.g., 1950 to 1987)
start_year = 1950
end_year = 1987
df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Extract the years and precipitation data
years = df_filtered['Year']
precipitation = df_filtered['Precipitation']

# Calculate the Sen's Slope
def sen_slope(x, y):
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            if x[j] - x[i] != 0:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)

    # Calculate the median of all slopes
    return np.median(slopes)

sen_sen_slope = sen_slope(years.tolist(), precipitation.tolist())

# Define a threshold to determine increasing or decreasing trend
threshold = 0.0

# Determine the trend direction
if sen_sen_slope > threshold:
    trend = "Increasing"
elif sen_sen_slope < -threshold:
    trend = "Decreasing"
else:
    trend = "Stable"

# Print the trend and Sen's Slope
print(f"Trend: {trend}")
print(f"Sen's Slope: {sen_sen_slope}")

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load your data into a Pandas DataFrame (replace 'your_data.csv' with your actual file)
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/timeseries_presample_LTP.xlsx")

# Filter data for the desired time range (1950-1987)
df_filtered = df[(df['Year'] >= 1988) & (df['Year'] <= 2022)]

# Perform linear regression
slope, intercept, r_value, p_value, std_err = linregress(df_filtered['Year'], df_filtered['Precipitation'])

# Calculate R-squared value
r_squared = r_value**2

# Create a bar chart
plt.figure(figsize=(10, 6))
plt.bar(df_filtered['Year'], df_filtered['Precipitation'], label="Precipitation")

# Add the R-squared value to the chart title
plt.title(f'Year-wise Precipitation (1988-2022)\nR-squared = {r_squared:.4f}')
plt.xlabel('Year')
plt.ylabel('Precipitation')
plt.legend()
plt.grid(axis='y')
plt.show()

import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats


# Read your time series data into a DataFrame
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/timeseries_presample_LTP.xlsx")

# Filter the data for the desired year range (e.g., 1950 to 1987)
start_year = 1988
end_year = 2022
df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Extract the years and precipitation data
years = df_filtered['Year']
precipitation = df_filtered['Precipitation']

# Calculate the Sen's Slope
def sen_slope(x, y):
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            if x[j] - x[i] != 0:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)

    # Calculate the median of all slopes
    return np.median(slopes)

sen_sen_slope = sen_slope(years.tolist(), precipitation.tolist())

# Define a threshold to determine increasing or decreasing trend
threshold = 0.0

# Determine the trend direction
if sen_sen_slope > threshold:
    trend = "Increasing"
elif sen_sen_slope < threshold:
    trend = "Decreasing"
else:
    trend = "Stable"

# Print the trend and Sen's Slope
print(f"Trend: {trend}")
print(f"Sen's Slope: {sen_sen_slope}")

import pandas as pd
import os

# Directory where your CSV files are located
data_dir = 'C:/5. Thesis/Prec Data/Prec_long term/Combined data'

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
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Yearwise_LTP.xlsx"

# Save the combined data to a new CSV file
combined_data.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats


# Read your time series data into a DataFrame
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Yearwise_LTP.xlsx")

# Filter the data for the desired year range (e.g., 1951 to 1987)
start_year = 1951
end_year = 1987
df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Extract the years and precipitation data
years = df_filtered['Year']
precipitation = df_filtered['Precipitation']

# Calculate the Sen's Slope
def sen_slope(x, y):
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            if x[j] - x[i] != 0:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)

    # Calculate the median of all slopes
    return np.median(slopes)

sen_sen_slope = sen_slope(years.tolist(), precipitation.tolist())

# Define a threshold to determine increasing or decreasing trend
threshold = 0.0

# Determine the trend direction
if sen_sen_slope > threshold:
    trend = "Increasing"
elif sen_sen_slope < threshold:
    trend = "Decreasing"
else:
    trend = "Stable"

# Print the trend and Sen's Slope
print(f"Trend: {trend}")
print(f"Sen's Slope: {sen_sen_slope}")

import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats


# Read your time series data into a DataFrame
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Yearwise_LTP.xlsx")

# Filter the data for the desired year range (e.g., 1950 to 1987)
start_year = 1988
end_year = 2022
df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Extract the years and precipitation data
years = df_filtered['Year']
precipitation = df_filtered['Precipitation']

# Calculate the Sen's Slope
def sen_slope(x, y):
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            if x[j] - x[i] != 0:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)

    # Calculate the median of all slopes
    return np.median(slopes)

sen_sen_slope = sen_slope(years.tolist(), precipitation.tolist())

# Define a threshold to determine increasing or decreasing trend
threshold = 0.0

# Determine the trend direction
if sen_sen_slope > threshold:
    trend = "Increasing"
elif sen_sen_slope < threshold:
    trend = "Decreasing"
else:
    trend = "Stable"

# Print the trend and Sen's Slope
print(f"Trend: {trend}")
print(f"Sen's Slope: {sen_sen_slope}")

import pandas as pd

# Load your data into a Pandas DataFrame (replace 'your_data.csv' with your actual file)
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Yearwise_LTP.xlsx")

# Group the data by year and sum the precipitation data for each year
aggregated_data = df.groupby('Year')['Precipitation'].sum().reset_index()

# Define the path to the new Excel file
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Yearwise_AggLTP.xlsx"

# Save the combined data to a new excel file
aggregated_data.to_excel(output_file, index=False)

import pandas as pd

# Read your original data file (replace 'your_data.csv' with your actual file path)
data = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Yearwise_LTP.xlsx")

# Group the data by year and calculate the mean of the precipitation data for each year
grouped_data = data.groupby('Year')['Precipitation'].agg(['sum', 'count']).reset_index()
grouped_data['Precipitation'] = grouped_data['sum'] / grouped_data['count']/349
grouped_data.drop(['sum', 'count'], axis=1, inplace=True)

# Define the path to the new Excel file
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Yearwise_AggLTP2.xlsx"

# Save the modified data to a new excel file
grouped_data.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, linregress

# Load your data (replace 'your_data.csv' with the actual file path)
file_path = 'C:/5. Thesis/Prec Data/Prec_long term/Analysis/Yearwise_AggLTP2.xlsx'
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
directory = "C:/5. Thesis/Prec Data/Prec_long term/Combined data/"

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
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Combined_data.xlsx"

# Save the data to a exce file
data.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
import os

# Define the directory where your 34 CSV files are located
directory = "C:/5. Thesis/Prec Data/Prec_long term/Combined data"

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
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Presample_Percentiles.xlsx"

# Save the aggregated data to a CSV file
aggregated_data.to_excel(output_file, index=False)

import pandas as pd

# Load your data into a DataFrame (assuming your data is in an Excel file)
data_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Presample_Percentiles.xlsx"
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
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Sample_Percentiles.xlsx"

# Save the aggregated data to the new Excel file
new_df.to_excel(output_file, index=False)

import numpy as np
import pandas as pd

# Read your data from the CSV file
data_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Sample_Percentiles.xlsx"
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
data_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Sample_Percentiles.xlsx"
df = pd.read_excel(data_file)

# Create a sample DataFrame with values greater than or equal to 8.070481709797825
sample_df = df[df['Precipitation'] >= 8.070481709797825]

# Define the path to save the sample
sample_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/75th Percentiles.xlsx"

# Save the sample DataFrame to a new CSV file
sample_df.to_excel(sample_file, index=False)

import pandas as pd

# Read your original data file 
data = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/75th Percentiles.xlsx")

# Group the data by year and calculate the mean of the precipitation data for each year
grouped_data = data.groupby('Year')['Precipitation'].agg(['sum', 'count']).reset_index()
grouped_data['Precipitation'] = grouped_data['sum'] / grouped_data['count']
grouped_data.drop(['sum', 'count'], axis=1, inplace=True)

# Define the path to the new Excel file
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/75th_Per_Yearwise.xlsx"

# Save the modified data to a new excel file
grouped_data.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, linregress

# Load your data (replace 'your_data.csv' with the actual file path)
file_path = 'C:/5. Thesis/Prec Data/Prec_long term/Analysis/75th Percentiles.xlsx'
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
data_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Sample_Percentiles.xlsx"
df = pd.read_excel(data_file)

# Create a sample DataFrame with values greater than or equal to 10.151885444469302
sample_df = df[df['Precipitation'] >= 10.151885444469302]

# Define the path to save the sample
sample_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/90th Percentiles.xlsx"

# Save the sample DataFrame to a new CSV file
sample_df.to_excel(sample_file, index=False)

import pandas as pd

# Read your original data file 
data = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/90th Percentiles.xlsx")

# Group the data by year and calculate the mean of the precipitation data for each year
grouped_data = data.groupby('Year')['Precipitation'].agg(['sum', 'count']).reset_index()
grouped_data['Precipitation'] = grouped_data['sum'] / grouped_data['count']
grouped_data.drop(['sum', 'count'], axis=1, inplace=True)

# Define the path to the new Excel file
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/90th_Per_Yearwise.xlsx"

# Save the modified data to a new excel file
grouped_data.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, linregress

# Load your data (replace 'your_data.csv' with the actual file path)
file_path = 'C:/5. Thesis/Prec Data/Prec_long term/Analysis/90th Percentiles.xlsx'
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
data_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Sample_Percentiles.xlsx"
df = pd.read_excel(data_file)

# Create a sample DataFrame with values greater than or equal to 11.579426849478217
sample_df = df[df['Precipitation'] >= 11.579426849478217]

# Define the path to save the sample
sample_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/95th Percentiles.xlsx"

# Save the sample DataFrame to a new CSV file
sample_df.to_excel(sample_file, index=False)

import pandas as pd

# Read your original data file 
data = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/95th Percentiles.xlsx")

# Group the data by year and calculate the mean of the precipitation data for each year
grouped_data = data.groupby('Year')['Precipitation'].agg(['sum', 'count']).reset_index()
grouped_data['Precipitation'] = grouped_data['sum'] / grouped_data['count']
grouped_data.drop(['sum', 'count'], axis=1, inplace=True)

# Define the path to the new Excel file
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/95th_Per_Yearwise.xlsx"

# Save the modified data to a new excel file
grouped_data.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, linregress

# Load your data (replace 'your_data.csv' with the actual file path)
file_path = 'C:/5. Thesis/Prec Data/Prec_long term/Analysis/95th Percentiles.xlsx'
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
import os

# Directory where your CSV files are located
data_dir = 'C:/5. Thesis/Prec Data/Prec_long term/Combined data'

# Initialize an empty DataFrame for combined data
combined_data = pd.DataFrame(columns=['Year', 'Month', 'Precipitation'])

# List all CSV files in the directory
csv_files = [file for file in os.listdir(data_dir) if file.endswith('.csv')]

# Define a mapping dictionary for month renaming
month_mapping = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

# Loop through each CSV file and aggregate precipitation data
for csv_file in csv_files:
    file_path = os.path.join(data_dir, csv_file)
    df = pd.read_csv(file_path)  # Read the CSV file
    
    # Exclude "Lat-Long" columns and sum the remaining columns
    precipitation_data = df.iloc[:, 4:].sum(axis=1)
    
    year_data = df['Year']
    month_data = df['Month']  # Extract the 'Month' column correctly
    
    # Rename the month values using the mapping dictionary
    month_data = month_data.map(month_mapping)
    
    # Combine data with the existing DataFrame
    combined_data = pd.concat([combined_data, pd.DataFrame({'Year': year_data, 'Month': month_data, 'Precipitation': precipitation_data})])

# Define the path to save the combined data to an Excel file
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Each year data/year_month_LTP.xlsx"

# Save the combined data to an Excel file
combined_data.to_excel(output_file, index=False)

import pandas as pd
import matplotlib.pyplot as plt

# Read your original data file (replace 'your_data.csv' with your actual file path)
data = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Each year data/year_month_LTP.xlsx")

# Get unique years in the data
unique_years = data['Year'].unique()

# Create a line chart for each year
for year in unique_years:
    year_data = data[data['Year'] == year]

    # Create a line chart for the current year
    plt.figure(figsize=(10, 6))
    plt.plot(year_data['Month'], year_data['Precipitation'], linestyle='-', marker='o', color='b')
    plt.xlabel('Month')
    plt.ylabel('Precipitation')
    plt.title(f'Precipitation for {year}')
    plt.grid(False)
    
    output_file = f"C:/5. Thesis/Prec Data/Prec_long term/Analysis/Each year data/Line Chart/precipitation_{year}.jpg"
    plt.savefig(output_file)
    
    # Close the current plot to release resources
    plt.close()
    

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit

# Read your original data file (replace 'your_data.csv' with your actual file path)
data = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Each year data/year_month_LTP.xlsx")

# Get unique years in the data
unique_years = data['Year'].unique()

# Create a line chart for each year
for year in unique_years:
    year_data = data[data['Year'] == year]

    # Create a line chart for the current year
    plt.figure(figsize=(10, 6))
    plt.bar(year_data['Month'], year_data['Precipitation'], color='b')
    plt.xlabel('Month')
    plt.ylabel('Precipitation')
    plt.title(f'Yearly Precipitation for {year}')
    plt.grid(False)
    
    output_file = f"C:/5. Thesis/Prec Data/Prec_long term/Analysis/Each year data/Bar Chart/precipitation_{year}.jpg"
    plt.savefig(output_file)
    
    # Close the current plot to release resources
    plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.polynomial import polyfit

# Read your original data file (replace 'your_data.csv' with your actual file path)
data = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Each year data/year_month_LTP.xlsx")

# Get unique years in the data
unique_years = data['Year'].unique()

# Create a line chart for each year
for year in unique_years:
    year_data = data[data['Year'] == year]

    # Create a line chart for the current year
    plt.figure(figsize=(10, 6))
    plt.plot(year_data['Precipitation'], color='b')
    plt.xlabel('Count')
    plt.ylabel('Precipitation')
    plt.title(f'Precipitation for {year}')
    plt.grid(False)
    
    output_file = f"C:/5. Thesis/Prec Data/Prec_long term/Analysis/Each year data/precipitation_{year}.jpg"
    plt.savefig(output_file)
    
    # Close the current plot to release resources
    plt.close()

import pandas as pd
import numpy as np
import os

# Define the directory where your files are located
directory = "C:/5. Thesis/Prec Data/Prec_long term/PP"

# Create a list of all precipitation file names
precip_file_names = [f for f in os.listdir(directory) if f[0].isdigit() and f.endswith(".csv")]

# Create a list of all date-time file names
datetime_file_names = [f for f in os.listdir(directory) if f.startswith("time") and f.endswith(".csv")]

# Loop through each pair of precipitation and date-time files
for precip_file_name, datetime_file_name in zip(precip_file_names, datetime_file_names):
    # Read the target location data (replace with the actual path)
    target_locations = pd.read_csv("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Jessore/cord_Jessore.csv")

    # Read the precipitation data and set the first column as the index
    daily_precip_data = pd.read_csv(os.path.join(directory, precip_file_name), header=0, index_col=0)

    # Extract the specific indexes from the target_locations DataFrame
    selected_indexes = target_locations['Order_']

    # Select the desired columns from the precipitation data using the selected indexes
    selected_precipitation_data = daily_precip_data.iloc[:, selected_indexes]

    # Define the path to the new CSV file for selected precipitation data
    new_precip_file_path = f"C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Jessore/fSorted Data_{precip_file_name}"

    # Save the selected precipitation data to the new file
    selected_precipitation_data.to_csv(new_precip_file_path)

    # Read the date and time data from the separate file
    date_time_data = pd.read_csv(os.path.join(directory, datetime_file_name))

    # Add a new column 'Identifier' to both DataFrames with the same values
    selected_precipitation_data['Identifier'] = np.arange(len(selected_precipitation_data))
    date_time_data.loc[:, 'Identifier'] = np.arange(len(date_time_data))

    # Combine the two DataFrames using 'Identifier' as the common column
    combined_data = pd.merge(date_time_data, selected_precipitation_data, on='Identifier')

    # Drop the 'Identifier' column if you no longer need it
    combined_data = combined_data.drop(columns=['Identifier'])

    # Define the column names (customize as needed)
    column_names = ['Year', 'Month', 'Date', 'Hour'] + [f'Lat_Long_{i}' for i in range(1, len(combined_data.columns) - 3)]

    # Assign the column names to the DataFrame
    combined_data.columns = column_names

    # Define the path to the new CSV file for combined data
    new_combined_file_path = f"C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Jessore/Jessore_{precip_file_name}"

    # Save the combined data to the new file
    combined_data.to_csv(new_combined_file_path, index=False)

print("All files processed successfully.")

import os
import pandas as pd
import numpy as np
import arcpy

# Define the path to your precipitation data file
directory = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Jessore"

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
data_dir = 'C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Jessore/Combined_Jessore'

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
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Jessore/Temporal Trend/Jessore_Yearwise_LTP.xlsx"

# Save the combined data to a new CSV file
combined_data.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats


# Read your time series data into a DataFrame
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Jessore/Temporal Trend/Jessore_Yearwise_LTP.xlsx")

# Filter the data for the desired year range (e.g., 1950 to 1987)
start_year = 1951
end_year = 1987
df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Extract the years and precipitation data
years = df_filtered['Year']
precipitation = df_filtered['Precipitation']

# Calculate the Sen's Slope
def sen_slope(x, y):
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            if x[j] - x[i] != 0:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)

    # Calculate the median of all slopes
    return np.median(slopes)

sen_sen_slope = sen_slope(years.tolist(), precipitation.tolist())

# Define a threshold to determine increasing or decreasing trend
threshold = 0.0

# Determine the trend direction
if sen_sen_slope > threshold:
    trend = "Increasing"
elif sen_sen_slope < threshold:
    trend = "Decreasing"
else:
    trend = "Stable"

# Print the trend and Sen's Slope
print(f"Trend: {trend}")
print(f"Sen's Slope: {sen_sen_slope}")

import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats


# Read your time series data into a DataFrame
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Jessore/Temporal Trend/Jessore_Yearwise_LTP.xlsx")

# Filter the data for the desired year range (e.g., 1950 to 1987)
start_year = 1988
end_year = 2022
df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Extract the years and precipitation data
years = df_filtered['Year']
precipitation = df_filtered['Precipitation']

# Calculate the Sen's Slope
def sen_slope(x, y):
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            if x[j] - x[i] != 0:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)

    # Calculate the median of all slopes
    return np.median(slopes)

sen_sen_slope = sen_slope(years.tolist(), precipitation.tolist())

# Define a threshold to determine increasing or decreasing trend
threshold = 0.0

# Determine the trend direction
if sen_sen_slope > threshold:
    trend = "Increasing"
elif sen_sen_slope < threshold:
    trend = "Decreasing"
else:
    trend = "Stable"

# Print the trend and Sen's Slope
print(f"Trend: {trend}")
print(f"Sen's Slope: {sen_sen_slope}")

import pandas as pd
import numpy as np
import os

# Define the directory where your files are located
directory = "C:/5. Thesis/Prec Data/Prec_long term/PP"

# Create a list of all precipitation file names
precip_file_names = [f for f in os.listdir(directory) if f[0].isdigit() and f.endswith(".csv")]

# Create a list of all date-time file names
datetime_file_names = [f for f in os.listdir(directory) if f.startswith("time") and f.endswith(".csv")]

# Loop through each pair of precipitation and date-time files
for precip_file_name, datetime_file_name in zip(precip_file_names, datetime_file_names):
    # Read the target location data (replace with the actual path)
    target_locations = pd.read_csv("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Khulna/cord_Khulna.csv")

    # Read the precipitation data and set the first column as the index
    daily_precip_data = pd.read_csv(os.path.join(directory, precip_file_name), header=0, index_col=0)

    # Extract the specific indexes from the target_locations DataFrame
    selected_indexes = target_locations['Order_']

    # Select the desired columns from the precipitation data using the selected indexes
    selected_precipitation_data = daily_precip_data.iloc[:, selected_indexes]

    # Define the path to the new CSV file for selected precipitation data
    new_precip_file_path = f"C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Khulna/Sorted Data/Sorted Data_{precip_file_name}"

    # Save the selected precipitation data to the new file
    selected_precipitation_data.to_csv(new_precip_file_path)

    # Read the date and time data from the separate file
    date_time_data = pd.read_csv(os.path.join(directory, datetime_file_name))

    # Add a new column 'Identifier' to both DataFrames with the same values
    selected_precipitation_data['Identifier'] = np.arange(len(selected_precipitation_data))
    date_time_data.loc[:, 'Identifier'] = np.arange(len(date_time_data))

    # Combine the two DataFrames using 'Identifier' as the common column
    combined_data = pd.merge(date_time_data, selected_precipitation_data, on='Identifier')

    # Drop the 'Identifier' column if you no longer need it
    combined_data = combined_data.drop(columns=['Identifier'])

    # Define the column names (customize as needed)
    column_names = ['Year', 'Month', 'Date', 'Hour'] + [f'Lat_Long_{i}' for i in range(1, len(combined_data.columns) - 3)]

    # Assign the column names to the DataFrame
    combined_data.columns = column_names

    # Define the path to the new CSV file for combined data
    new_combined_file_path = f"C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Khulna/Combined_Khulna/Khulna_{precip_file_name}"

    # Save the combined data to the new file
    combined_data.to_csv(new_combined_file_path, index=False)

print("All files processed successfully.")

import os
import pandas as pd
import numpy as np
import arcpy

# Define the path to your precipitation data file
directory = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Khulna/Combined_Khulna"

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
data_dir = 'C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Khulna/Combined_Khulna'

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
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Khulna/Temporal Trend/Khulna_Yearwise_LTP.xlsx"

# Save the combined data to a new CSV file
combined_data.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats


# Read your time series data into a DataFrame
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Khulna/Temporal Trend/Khulna_Yearwise_LTP.xlsx")

# Filter the data for the desired year range (e.g., 1950 to 1987)
start_year = 1950
end_year = 1987
df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Extract the years and precipitation data
years = df_filtered['Year']
precipitation = df_filtered['Precipitation']

# Calculate the Sen's Slope
def sen_slope(x, y):
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            if x[j] - x[i] != 0:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)

    # Calculate the median of all slopes
    return np.median(slopes)

sen_sen_slope = sen_slope(years.tolist(), precipitation.tolist())

# Define a threshold to determine increasing or decreasing trend
threshold = 0.0

# Determine the trend direction
if sen_sen_slope > threshold:
    trend = "Increasing"
elif sen_sen_slope < threshold:
    trend = "Decreasing"
else:
    trend = "Stable"

# Print the trend and Sen's Slope
print(f"Trend: {trend}")
print(f"Sen's Slope: {sen_sen_slope}")

import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats


# Read your time series data into a DataFrame
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Khulna/Temporal Trend/Khulna_Yearwise_LTP.xlsx")

# Filter the data for the desired year range (e.g., 1950 to 1987)
start_year = 1988
end_year = 2022
df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Extract the years and precipitation data
years = df_filtered['Year']
precipitation = df_filtered['Precipitation']

# Calculate the Sen's Slope
def sen_slope(x, y):
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            if x[j] - x[i] != 0:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)

    # Calculate the median of all slopes
    return np.median(slopes)

sen_sen_slope = sen_slope(years.tolist(), precipitation.tolist())

# Define a threshold to determine increasing or decreasing trend
threshold = 0.0

# Determine the trend direction
if sen_sen_slope > threshold:
    trend = "Increasing"
elif sen_sen_slope < threshold:
    trend = "Decreasing"
else:
    trend = "Stable"

# Print the trend and Sen's Slope
print(f"Trend: {trend}")
print(f"Sen's Slope: {sen_sen_slope}")

import pandas as pd
import numpy as np
import os

# Define the directory where your files are located
directory = "C:/5. Thesis/Prec Data/Prec_long term/PP"

# Create a list of all precipitation file names
precip_file_names = [f for f in os.listdir(directory) if f[0].isdigit() and f.endswith(".csv")]

# Create a list of all date-time file names
datetime_file_names = [f for f in os.listdir(directory) if f.startswith("time") and f.endswith(".csv")]

# Loop through each pair of precipitation and date-time files
for precip_file_name, datetime_file_name in zip(precip_file_names, datetime_file_names):
    # Read the target location data (replace with the actual path)
    target_locations = pd.read_csv("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Satkhira/cord_Satkhira.csv")

    # Read the precipitation data and set the first column as the index
    daily_precip_data = pd.read_csv(os.path.join(directory, precip_file_name), header=0, index_col=0)

    # Extract the specific indexes from the target_locations DataFrame
    selected_indexes = target_locations['Order_']

    # Select the desired columns from the precipitation data using the selected indexes
    selected_precipitation_data = daily_precip_data.iloc[:, selected_indexes]

    # Define the path to the new CSV file for selected precipitation data
    new_precip_file_path = f"C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Satkhira/Sorted Data/Sorted Data_{precip_file_name}"

    # Save the selected precipitation data to the new file
    selected_precipitation_data.to_csv(new_precip_file_path)

    # Read the date and time data from the separate file
    date_time_data = pd.read_csv(os.path.join(directory, datetime_file_name))

    # Add a new column 'Identifier' to both DataFrames with the same values
    selected_precipitation_data['Identifier'] = np.arange(len(selected_precipitation_data))
    date_time_data.loc[:, 'Identifier'] = np.arange(len(date_time_data))

    # Combine the two DataFrames using 'Identifier' as the common column
    combined_data = pd.merge(date_time_data, selected_precipitation_data, on='Identifier')

    # Drop the 'Identifier' column if you no longer need it
    combined_data = combined_data.drop(columns=['Identifier'])

    # Define the column names (customize as needed)
    column_names = ['Year', 'Month', 'Date', 'Hour'] + [f'Lat_Long_{i}' for i in range(1, len(combined_data.columns) - 3)]

    # Assign the column names to the DataFrame
    combined_data.columns = column_names

    # Define the path to the new CSV file for combined data
    new_combined_file_path = f"C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Satkhira/Combined_Satkhira/Satkhira_{precip_file_name}"

    # Save the combined data to the new file
    combined_data.to_csv(new_combined_file_path, index=False)

print("All files processed successfully.")

import os
import pandas as pd
import numpy as np
import arcpy

# Define the path to your precipitation data file
directory = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Satkhira/Combined_Satkhira"

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
data_dir = 'C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Satkhira/Combined_Satkhira'

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
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Satkhira/Temporal Trend/Satkhira_Yearwise_LTP.xlsx"

# Save the combined data to a new CSV file
combined_data.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats


# Read your time series data into a DataFrame
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Satkhira/Temporal Trend/Satkhira_Yearwise_LTP.xlsx")

# Filter the data for the desired year range (e.g., 1950 to 1987)
start_year = 1950
end_year = 1987
df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Extract the years and precipitation data
years = df_filtered['Year']
precipitation = df_filtered['Precipitation']

# Calculate the Sen's Slope
def sen_slope(x, y):
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            if x[j] - x[i] != 0:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)

    # Calculate the median of all slopes
    return np.median(slopes)

sen_sen_slope = sen_slope(years.tolist(), precipitation.tolist())

# Define a threshold to determine increasing or decreasing trend
threshold = 0.0

# Determine the trend direction
if sen_sen_slope > threshold:
    trend = "Increasing"
elif sen_sen_slope < threshold:
    trend = "Decreasing"
else:
    trend = "Stable"

# Print the trend and Sen's Slope
print(f"Trend: {trend}")
print(f"Sen's Slope: {sen_sen_slope}")

import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy import stats


# Read your time series data into a DataFrame
df = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Spatial Data/Satkhira/Temporal Trend/Satkhira_Yearwise_LTP.xlsx")

# Filter the data for the desired year range (e.g., 1950 to 1987)
start_year = 1988
end_year = 2022
df_filtered = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

# Extract the years and precipitation data
years = df_filtered['Year']
precipitation = df_filtered['Precipitation']

# Calculate the Sen's Slope
def sen_slope(x, y):
    n = len(x)
    slopes = []

    for i in range(n):
        for j in range(i + 1, n):
            if x[j] - x[i] != 0:
                slope = (y[j] - y[i]) / (x[j] - x[i])
                slopes.append(slope)

    # Calculate the median of all slopes
    return np.median(slopes)

sen_sen_slope = sen_slope(years.tolist(), precipitation.tolist())

# Define a threshold to determine increasing or decreasing trend
threshold = 0.0

# Determine the trend direction
if sen_sen_slope > threshold:
    trend = "Increasing"
elif sen_sen_slope < threshold:
    trend = "Decreasing"
else:
    trend = "Stable"

# Print the trend and Sen's Slope
print(f"Trend: {trend}")
print(f"Sen's Slope: {sen_sen_slope}")

import pandas as pd

# Read your original data file (replace 'your_data.csv' with your actual file path)
data = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Yearwise_LTP.xlsx")

# Group the data by year and calculate the mean of the precipitation data for each year
grouped_data = data.groupby('Year')['Precipitation'].agg(['sum', 'count']).reset_index()
grouped_data['Precipitation'] = grouped_data['sum'] / grouped_data['count']
grouped_data.drop(['sum', 'count'], axis=1, inplace=True)

# Define the path to the new Excel file
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Accum.xlsx"

# Save the modified data to a new excel file
grouped_data.to_excel(output_file, index=False)

import pandas as pd

# Load your data into DataFrames
accumulated_precip_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/Accum.xlsx"
extreme_std_file = "C:/5. Thesis/Prec Data/Prec_long term/Combined data/combined_2000.csv"

# Read accumulated precipitation data
accumulated_df = pd.read_excel(accumulated_precip_file)

# Read extreme precipitation and standard deviation data
extreme_std_df = pd.read_csv(extreme_std_file)

# Assuming the columns for year, extreme precip, and std in extreme_std_df are named appropriately
# If not, replace 'Year', 'Extreme Precipitation', and 'Standard Deviation' with your actual column names
extreme_std_df = extreme_std_df.rename(columns={'Year': 'Year', 'Extreme Precipitation': 'Extreme Precipitation', 'Standard Deviation': 'Standard Deviation'})

# Merge the two DataFrames on the 'Year' column
result_df = pd.merge(accumulated_df, extreme_std_df, on='Year')

# Display or save the results as needed
print(result_df)

# Save the combined DataFrame to a new file
result_df.to_csv("path_to_output_file.csv", index=False)

import pandas as pd

# Load your data into DataFrames
extreme_std_file = "C:/5. Thesis/Prec Data/Prec_long term/Combined data/combined_2008.csv"

# Read extreme precipitation and standard deviation data
extreme_std_df = pd.read_csv(extreme_std_file)

# Subset to just precipitation columns
precip_cols = df.columns[4:] 

# Flatten entire dataframe to one array
flat_precip = df[precip_cols].replace(np.nan, 0).values.flatten()

# Calculate statistics on flat array
ext_precip = flat_precip.max()  
std_dev = flat_precip.std()   

# Create stats dataframe
stats_df = pd.DataFrame({
    "Extreme Precipitation": [ext_precip], 
    "Standard Deviation": [std_dev]
}, index=[0]) 

print(stats_df)

import pandas as pd
import numpy as np
import os

# Define the directory where your 34 CSV files are located
directory = "C:/5. Thesis/Prec Data/Prec_long term/Combined data"

# Get a list of all CSV files in the directory
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]

# Initialize a DataFrame to store the aggregated results
aggregated_data = pd.DataFrame()

# Loop through each CSV file
for csv_file in csv_files:
    # Read the CSV file into a DataFrame
    df = pd.read_csv(os.path.join(directory, csv_file))

    # Sum the precipitation for each column except Month, Date, and Hour
    sum_precip = df.drop(columns=["Hour"]).groupby(['Year', 'Month', 'Date']).sum().reset_index()

    # Add the result to the aggregated DataFrame
    aggregated_data = pd.concat([aggregated_data, sum_precip], ignore_index=True)

# Define the path to save the aggregated data
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/MAvg_pre.xlsx"

# Save the aggregated data to an Excel file
aggregated_data.to_excel(output_file, index=False)

import pandas as pd

# Read your original data file (replace 'your_data.xlsx' with your actual file path)
data = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/MAvg_pre.xlsx")

# Subset to just precipitation columns
precip_cols = data.columns[3:] 

# Sum the precipitation data from Lat_Long_1 to Lat_Long_349
data['Precipitation'] = data[precip_cols].sum(axis=1)  # Summing precipitation values

# Divide the aggregated precipitation values by 349
data['Precipitation'] /= 349  # Divide by 349

# Selecting necessary columns to retain (Year, Month, Date, and adjusted aggregated precipitation values)
selected_cols = ['Year', 'Month', 'Date', 'Precipitation']
aggregated_data = data[selected_cols].copy()

# Display a sample of the aggregated data
print(aggregated_data.head())

# Define the path to save the aggregated data
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/MAvg_sample.xlsx"  # Replace this with your desired output file path

# Save the aggregated data to an Excel file
aggregated_data.to_excel(output_file, index=False)

import matplotlib.pyplot as plt

file_path = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/MAvg_sample.xlsx"

# Read data from the Excel file into a DataFrame
aggregated_data = pd.read_excel(file_path)

# Sort the DataFrame by date to ensure the rolling window operates correctly
aggregated_data = aggregated_data.sort_values(by=['Year', 'Month', 'Date'])

# Calculate the 5-day moving average for the 'Precipitation' column
aggregated_data['5D_Moving_Avg'] = aggregated_data['Precipitation'].rolling(window=5, min_periods=1).mean()

# Display a sample of the DataFrame with the added 5-day moving average column
print(aggregated_data.head())

# Or to view only specific columns (Year, Month, Date, Precipitation, 5-day moving average)
print(aggregated_data[['Year', 'Month', 'Date', 'Precipitation', '5D_Moving_Avg']].head())

# Plot the time series of daily precipitation
plt.figure(figsize=(12, 6))
plt.plot(aggregated_data["Date"], aggregated_data["Precipitation"], label="Daily Precipitation", color='royalblue')
plt.xlabel("Date")
plt.ylabel("Precipitation")
plt.title("Daily Precipitation Time Series")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Calculate and plot the 5-day moving average precipitation
moving_avg_5d = aggregated_data["Precipitation"].rolling(window=5).mean()

plt.figure(figsize=(12, 6))
plt.plot(aggregated_data["Date"], moving_avg_5d, label="5-Day Moving Average Precipitation", color='orange')
plt.xlabel("Date")
plt.ylabel("Precipitation")
plt.title("5-Day Moving Average Precipitation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Analyze seasonal decomposition for precipitation (assuming a subset of data for faster computation)
subset_data = aggregated_data[:365 * 5]  # Using the first 5 years for decomposition

result = seasonal_decompose(subset_data["Precipitation"], model='additive', period=365)
result.plot()
plt.suptitle('Seasonal Decomposition of Precipitation')
plt.tight_layout()
plt.show()

import pandas as pd

# Read your original data file (replace 'your_data.xlsx' with your actual file path)
data = pd.read_excel("C:/5. Thesis/Prec Data/Prec_long term/Analysis/Yearwise_AggLTP2.xlsx")

# Calculate the 5-year moving average for precipitation
moving_avg_5y = data['Precipitation'].rolling(window= 5).mean()

# Create a new DataFrame to store the Date and 5-year moving average
moving_avg_df = pd.DataFrame({
    'Date': data['Year'],  # Replace 'Date' with your actual date column name
    '5D_Moving_Avg_Precipitation': moving_avg_5y
})

# Save the 5-year moving average data to a new CSV file
output_file = "C:/5. Thesis/Prec Data/Prec_long term/Analysis/MAvg.xlsx"  # Replace with your desired file path
moving_avg_df.to_excel(output_file, index=False)

import pandas as pd
import numpy as np
from scipy.stats import kendalltau, linregress

# Load your data (replace 'your_data.csv' with the actual file path)
file_path = 'C:/5. Thesis/Prec Data/Prec_long term/Analysis/MAvg.xlsx'
data = pd.read_excel(file_path)

# Extract year and precipitation columns
years = data['Date']
precipitation = data['5Y_Moving_Avg_Precipitation']

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


