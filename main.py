import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from libpysal.weights import Queen
from esda.moran import Moran
from pysal.model import spreg
import numpy as np

####################

# 1. Load and Visualize the Dataset

shapefile_path = "States_shapefile-shp/States_shapefile.shp"
shapefile = gpd.read_file(shapefile_path)

# Load socio-economic data from CSV files
income_df = pd.read_csv("income.csv", header=None, names=["Location", "Median_Annual_Household_Income"])
poverty_df = pd.read_csv("poverty.csv", header=None, names=["Location", "Poverty_Rate"])
unemployment_df = pd.read_csv("unemp.csv", header=None, names=["Location", "Unemployment_Rate"])
# print(income_df)

shapefile['State_Name_Lower'] = shapefile['State_Name'].str.lower()
income_df['Location_Lower'] = income_df['Location'].str.lower()
poverty_df['Location_Lower'] = poverty_df['Location'].str.lower()
unemployment_df['Location_Lower'] = unemployment_df['Location'].str.lower()

# Merge shapefile with socio-economic data

merged_data = shapefile.merge(income_df, how='left', left_on='State_Name_Lower', right_on='Location_Lower')
merged_data = merged_data.merge(poverty_df, how='left', left_on='State_Name_Lower', right_on='Location_Lower')
merged_data = merged_data.merge(unemployment_df, how='left', left_on='State_Name_Lower', right_on='Location_Lower')


# Spatial plot

'''
shapefile.plot()
plt.title('Spatial Distribution of States')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

'''

# Income plot
'''
merged_data.plot(column='Median_Annual_Household_Income', cmap='Blues', legend=True)

plt.title('Spatial Distribution of Median Annual Household Income')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

'''
# Poverty plot
'''
merged_data.plot(column='Poverty_Rate', cmap='Reds', legend=True)

plt.title('Spatial Distribution of Poverty Rate')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

'''
# Unemployment plot

'''
merged_data.plot(column='Unemployment_Rate', cmap='Greens', legend=True)

plt.title('Spatial Distribution of Unemployment Rate')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

'''

####################

# 2. Spatial weights matrix

merged_data_filtered = merged_data[merged_data['State_Name_Lower'] != 'alaska']
merged_data_filtered = merged_data_filtered[merged_data_filtered['State_Name_Lower'] != 'hawaii']

# Create spatial weights matrix using Queen contiguity
w = Queen.from_dataframe(merged_data_filtered)
 
# Print the spatial weights matrix
print(w)

####################

# 3. Spatial Autocorrelation 

# Calculate Moran's I for a socio-economic indicator (e.g., income levels)
y = merged_data_filtered['Median_Annual_Household_Income'] = pd.to_numeric(merged_data_filtered['Median_Annual_Household_Income'].str.replace('$', '').str.replace(',', ''))

moran = Moran(y, w)

# Print Moran's I statistic
# print("Moran's I:", moran.I)

####################

# 4. Spatial Regression


# Prepare data
y = np.array(merged_data_filtered['Median_Annual_Household_Income'])
X = np.array(merged_data_filtered[['Poverty_Rate', 'Unemployment_Rate']])  

# Create spatial weights matrix
w = Queen.from_dataframe(merged_data_filtered)

# Run Spatial Lag Model
model = spreg.ML_Lag(y, X, w=w, name_y='Median_Annual_Household_Income', name_x=['Poverty_Rate', 'Unemployment_Rate']) 
# print(model.summary)


####################

# 5. Visualization

# Plot choropleth map
merged_data_filtered.plot(column='Median_Annual_Household_Income', cmap='Blues', legend=True)
plt.title('Median Annual Household Income')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()