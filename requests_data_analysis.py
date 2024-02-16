#!/usr/bin/env python
# coding: utf-8

# # Point72 MI Data - Analytics Case Study
# 
# ### Introduction
# 
# Shown below is the problem statement of this case study. 
# 
# * You are given two types of data: 311 Service requests, and weather. The 311 Service requests dataset contains all the 311 calls from 2016 to 2018 with many different features including timestamps, location, service request descriptions, etc. Please describe the steps you take to ingest and process the data, what are the challenges and how you overcome them.
# * Aggregate the 311 records in ways that best describes the characteristics of the inbound call patterns. Explain and visualize your findings. What are the implications you can infer from the patterns?
# * The weather data contains the weather information from the weather stations in NYC from 2010 to 2018 in daily intervals. Describe and visualize your findings. (Please note this is NOT weather forecast)
# * The final goal of this project is to predict the daily 311 inbound calls for the next 7 days 
# * Use your insights from previous sections to build a predictive model of your choice – explain the assumptions of your model and why you picked it.
# * What features are you using in your model and, how does your data exploration process help you decide which features you are incorporating in your model?
# * Evaluate the impact of weather on the call volume, find patterns and define causal impact if there’s any. 
# * Build a reasonable model framework, explain your model results, and justify the relationships identified in the model. Not necessary to dig into complex models.
# 
# 

# ### 311 requests
# 
# First, we'll look at the 311 calls dataset. This dataset in its raw form is quite large (7M rows, and 43 columns). To avoid running into memory issues, and keep runtimes reasonable, we can sample a chunk of the dataset instead of reading all of it. We can increase the sample size after the data exploration phase, if a need arises - such as if an analysis we're interested in requires more data points. 
# 
# 
# To analyze the 311 Service requests dataset, we will follow a structured approach to ingest, process, and aggregate the data, focusing on understanding the inbound call patterns. Here's an outline of the steps we'll take, the potential challenges we might encounter, and how we'll overcome them, followed by an analysis and visualization of our findings.
# 
# #### Step 1: Data Ingestion
# **Action**: Load the dataset using Pandas.\
# **Challenge**: The file size is very large, and leads to memory issues.\
# **Solution**: Use chunking and randomly sample data points so we have span all the timeperiods.
# 
# #### Step 2: Preliminary Exploration
# **Action**: Perform an initial exploration to understand the dataset's structure, including the number of records, features, missing values, and data types.\
# **Challenge**: Identifying relevant features and handling missing or inconsistent data.\
# **Solution**: Use descriptive statistics and visualization tools to assess data quality, and apply imputation or removal of missing data as appropriate.
# 
# #### Step 3: Data Cleaning
# **Action**: Clean the dataset by handling missing values, correcting data types (e.g., converting timestamps to datetime objects), and removing duplicates.\
# **Challenge**: Ensuring accurate data type conversions and dealing with outliers.\
# **Solution**: Validate conversions through sample checks and use statistical methods to identify and handle outliers.
# 
# #### Step 4: Data Aggregation
# **Action**: Aggregate the data to identify patterns, such as call volume over time, most common service requests, and geographic distribution of calls.\
# **Challenge**: Choosing the right level of aggregation to reveal meaningful patterns without oversimplification.\
# **Solution**: Experiment with different aggregation levels (e.g., daily, monthly, by neighborhood) and metrics (e.g., count, mean) to find the most insightful views.
# 
# #### Step 5: Analysis and Visualization
# **Action**: Analyze aggregated data to uncover trends, seasonal patterns, and anomalies. Visualize findings using charts and maps.\
# **Challenge**: Making complex data understandable and visually engaging.\
# **Solution**: Use a combination of visualization techniques, such as time series plots, bar charts, heatmaps, and geographic maps.
# 
# #### Step 6: Implications and Insights
# **Action**: Interpret patterns to infer implications for city management, resource allocation, and policy making.\
# **Challenge**: Translating data patterns into actionable insights.\
# **Solution**: Combine data analysis with domain knowledge to provide recommendations or insights.
# Let's start by ingesting the dataset and performing a preliminary exploration to understand its structure. We'll look at the number of records, features, and get a sense of the data we're dealing with. Then, we'll proceed with the subsequent steps based on our initial findings.
# 
# 
# 
# We'll jump right into it by importing libraries. Then we'll load and examine the data to understand its structure and contents. This examination will allow us to identify the best ways to aggregate the 311 records and describe the characteristics of the inbound call patterns. Let's start by loading the data and taking a look at the first few rows.

# In[158]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from io import StringIO
import random
import plotly.express as px

font = {'size'   : 18}
plt.rc('font', **font)


# In[159]:


filename = '311-2016-2018.csv'

n = sum(1 for line in open(filename)) - 1 # number of rows in the original dataset
s = 1000000 # we want to sample 500k rows, randomly
skip = sorted(random.sample(range(1,n+1),n-s)) # the sampling step

requests = pd.read_csv(filename, skiprows=skip, on_bad_lines='skip')
requests['Created Date'] = pd.to_datetime(requests['Created Date'])
requests.head()


# In[160]:


requests.columns


# The dataset contains various columns, including identifiers, dates, agency details, complaint types, descriptors, location information, and more. Here's a brief overview based on the first few rows:
# 
# * Unique Key: A unique identifier for each record.
# * Created Date: The date and time when the complaint was created.
# * Closed Date: The date and time when the complaint was closed.
# * Agency: The acronym of the agency responsible for addressing the complaint.
# * Agency Name: The full name of the agency.
# * Complaint Type: The type of complaint.
# * Descriptor: More detailed information about the complaint.
# * Location Type: The type of location where the incident was reported.
# * Incident Zip: The ZIP code where the incident occurred.
# * Latitude, Longitude, Location: Geographic details of the incident location.
# * Created Year and Date: The year and date when the complaint was created, respectively.

# ### Exploratory data analysis
# 
# To describe the characteristics of the inbound call patterns, we can aggregate the data in several ways:
# 
# 1. Time Trends: Analyze complaint volume over time (daily, monthly, yearly).
# 2. Complaint Type Distribution: Identify the most common types of complaints.
# 3. Agency Response: Examine which agencies handle the most complaints.
# 4. Location Analysis: Determine areas with high complaint volumes.
# 5. Resolution Time: Calculate the time taken to close complaints.
# 
# For a comprehensive analysis, we'll start by exploring each of these aggregations. 
# 
# #### Yearly, monthly trends
# 
# Let's begin with the time trends to see how complaint volumes have changed over time. We'll look at the yearly and monthly complaint volumes.

# In[161]:


# Extract year and month for aggregation
requests['Year'] = requests['Created Date'].dt.year
requests['Month'] = requests['Created Date'].dt.month

# Yearly complaint volumes
yearly_volume = requests.groupby('Year').size()

# Monthly complaint volumes (across all years)
monthly_volume = requests.groupby('Month').size()

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(12, 12))

# Yearly Volumes
yearly_volume.plot(kind='bar', ax=ax[0], color='skyblue')
ax[0].set_title('Yearly Complaint Volumes')
ax[0].set_xlabel('Year')
ax[0].set_ylabel('Number of Complaints')
ax[0].grid()

# Monthly Volumes
monthly_volume.plot(kind='bar', ax=ax[1], color='lightgreen')
ax[1].set_title('Monthly Complaint Volumes (Across All Years)')
ax[1].set_xlabel('Month')
ax[1].set_ylabel('Number of Complaints')
ax[1].grid()

plt.tight_layout()
plt.show()


# In[162]:


# Extract year and month for grouping purposes
requests['Year-Month'] = requests['Created Date'].dt.to_period('M')

# Group by the new 'Year-Month' column to see the number of requests over time
requests_by_month = requests.groupby('Year-Month').size()

# Plotting
plt.figure(figsize=(14, 7))
requests_by_month.plot(kind='line', color='blue', marker='o')
plt.title('311 Requests Over Time')
plt.xlabel('Year and Month')
plt.ylabel('Number of Requests')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()


# The visualizations provide insights into the inbound call patterns based on the 311 records:
# 
# 1. **Yearly Complaint Volumes**: The bar chart shows the number of complaints for each year represented in the dataset. It looks like the number of complaints have increased over the 3 years. 
# 
# 2. **Monthly Complaint Volumes (Across All Years)**: The second bar chart aggregates complaint volumes by month, combining data from all years. Not really seeing any major trends here - this is somewhat surprising since I expected seasonal trends. 

# In[163]:


# Extract the week of the year from the 'Request Date'
requests['Request Week'] = requests['Created Date'].dt.isocalendar().week

# Group by year as well to ensure accuracy across years
requests['Request Year'] = requests['Created Date'].dt.year

# Aggregate the data by year and week
weekly_requests = requests.groupby(['Request Year', 'Request Week']).size()

# Reset index to have a flat structure for easier plotting
weekly_requests_reset = weekly_requests.reset_index(name='Count')

# Plotting the weekly request counts
plt.figure(figsize=(15, 6))
plt.plot(weekly_requests_reset.index, weekly_requests_reset['Count'], marker='o', linestyle='-', markersize=5, alpha=0.75, color='green')
plt.title('311 Requests Count per Week')
plt.xlabel('Week Number')
plt.ylabel('Number of Requests')
plt.grid(axis='y', linestyle='--')

plt.show()


# 
# 
# #### Complaint Type Distribution
# We'll next look at the distribution of complaint types. This will help us understand the most common issues reported by residents. After that, we can explore the resolution times to gauge the efficiency of the agencies' responses.
# 

# In[164]:


complaint_type_distribution = requests['Complaint Type'].value_counts().head(20)

# Plotting
font = {'size'   : 15}
plt.rc('font', **font)

plt.figure(figsize=(10, 8))
complaint_type_distribution.plot(kind='barh', color='cadetblue')
plt.title('Top 20 Complaint Types')
plt.xlabel('Number of Complaints')
plt.ylabel('Complaint Type')
plt.gca().invert_yaxis()  # To display the highest number at the top
plt.grid()
plt.show()


# The bar chart displays the top 20 complaint types based on their frequencies. This visualization helps us identify the most common issues that residents' report. As we can see, **Noise - Residential** and **Heat/Hot Water** are the most frequent types of requests. 
# 
# #### Agency Name Distribution
# We'll next look at the distribution of frequency of complaints by agency. 
# 

# In[165]:


agency_name_distribution = requests['Agency Name'].value_counts().head(20)

plt.figure(figsize=(12, 8))
agency_name_distribution.plot(kind='barh', color='cadetblue')
plt.title('Top 20 Agencies')
plt.xlabel('Number of Complaints')
plt.ylabel('Agency Name')
plt.gca().invert_yaxis()  # To display the highest number at the top
plt.grid()
plt.show()


# We see that NYCPD and Department of Housing Preservation and Development receive the highest number of complaints. 
# 
# #### Which Borough has the highest requests? 
# 

# In[166]:


import matplotlib.pyplot as plt

# Count the number of complaints/requests from different boroughs
borough_counts = requests['Borough'].value_counts()

# Plotting the distribution of complaints/requests across different boroughs
plt.figure(figsize=(10, 6))
borough_counts.plot(kind='bar')
plt.title('Number of 311 Requests by Borough')
plt.xlabel('Borough')
plt.ylabel('Number of Requests')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')

plt.show()


# Given that the different boroughs have different populations, we can get the per capita complaint frequency by normalizing the above counts by the population. The population of different Boroughs from a recent census is as follows -
# 
# Brooklyn: 2.6 million
# Queens: 2.3 million
# Manhattan: 1.6 million
# The Bronx: 1.4 million
# Staten Island: 0.5 million
# 
# Next, we'll calculate the per capita complaints/requests for each borough by dividing the total number of complaints/requests from each borough by its population.
# 
# 

# In[167]:


# Correcting the calculation for per capita requests

# Population data for each borough
populations = {
    'BROOKLYN': 2.6e6,
    'QUEENS': 2.3e6,
    'MANHATTAN': 1.6e6,
    'BRONX': 1.4e6,
    'STATEN ISLAND': 0.5e6
}


# First, ensure we correctly aggregate the number of requests per borough according to the updated population keys
borough_counts_corrected = requests['Borough'].value_counts()

# Calculate the per capita complaints/requests for each borough using the corrected aggregation
per_capita_requests_corrected = {borough: borough_counts_corrected[borough] / populations[borough] for borough in populations.keys()}

per_capita_requests_corrected_df = pd.Series(per_capita_requests_corrected).sort_values(ascending=False)

# Plotting the per capita complaints/requests by borough
plt.figure(figsize=(10, 6))
per_capita_requests_corrected_df.plot(kind='bar', color='skyblue')
plt.title('311 Requests per Capita by Borough')
plt.xlabel('Borough')
plt.ylabel('Requests per Capita')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')

plt.show()


# #### Geographical locations of requests 
# Since we have the latitude, longitude of the reported incident, we can plot the locations and density on a map of New York (mostly because it's cool!). 

# In[168]:


requests_lat_long = requests.copy()

requests_lat_long['Latitude'] = np.round(requests_lat_long['Latitude'],2)
requests_lat_long['Longitude'] = np.round(requests_lat_long['Longitude'],2)

requests_lat_long_agg = requests_lat_long.groupby(['Latitude','Longitude']).size().reset_index(name='counts')
requests_lat_long_agg['Density'] = requests_lat_long_agg['counts']/requests_lat_long_agg['counts'].sum()

color_scale = [(0, 'orange'), (1,'red')]

fig = px.density_mapbox(requests_lat_long_agg, 
                        lat="Latitude", 
                        lon="Longitude", 
                        z="Density",
                        color_continuous_scale='viridis',
                        zoom=9, 
                        radius=26,
                        height=800,
                        width=900)

fig.update_layout(mapbox_style="open-street-map")
fig.show()


# In[169]:


requests['Created Date'] = pd.to_datetime(requests['Created Date'], errors='coerce')
requests['Closed Date'] = pd.to_datetime(requests['Closed Date'], errors='coerce')

# Calculate resolution time in days
requests['Resolution Time'] = (requests['Closed Date'] - requests['Created Date']).dt.total_seconds() / (60 * 60 * 24)

# Group by 'Complaint Type' and calculate average resolution time
average_resolution_time = requests.groupby('Complaint Type')['Resolution Time'].mean().reset_index()

average_resolution_time.sort_values(by='Resolution Time', ascending=False)


# In[170]:


# Since the list is long, let's take the top 20 complaint types for a clearer visualization
top_20_average_resolution_time = average_resolution_time.nlargest(20, 'Resolution Time')

# Plotting
plt.figure(figsize=(12, 8))
plt.barh(top_20_average_resolution_time['Complaint Type'], top_20_average_resolution_time['Resolution Time'], color='skyblue')
plt.xlabel('Average Resolution Time (Days)')
plt.ylabel('Complaint Type')
plt.title('Top 20 Complaint Types by Average Resolution Time')
plt.gca().invert_yaxis()  # To display the longest resolution time at the top
plt.tight_layout()
plt.grid()
plt.show()


# In[171]:


# Since the list is long, let's take the top 20 complaint types for a clearer visualization
bottom_20_average_resolution_time = average_resolution_time.dropna().nsmallest(20, 'Resolution Time')

# Plotting
plt.figure(figsize=(12, 8))
plt.barh(bottom_20_average_resolution_time['Complaint Type'], bottom_20_average_resolution_time['Resolution Time'], color='skyblue')
plt.xlabel('Average Resolution Time (Days)')
plt.ylabel('Complaint Type')
plt.title('Bottom 20 Complaint Types by Average Resolution Time')
plt.gca().invert_yaxis()  # To display the longest resolution time at the top
plt.tight_layout()
plt.grid()
plt.show()


# In[172]:


# Filter out records with negative resolution times
data_corrected = requests[requests['Resolution Time'] >= 0]

# Re-calculate the average resolution time by complaint type, focusing on corrected data
average_resolution_time_corrected = data_corrected.groupby('Complaint Type')['Resolution Time'].mean().sort_values(ascending=True).head(10)

# Plotting the corrected average resolution times
plt.figure(figsize=(10, 6))
average_resolution_time_corrected.plot(kind='barh', color='lightblue')
plt.title('Top 10 Complaint Types with Fastest Average Resolution Time (Corrected)')
plt.xlabel('Average Resolution Time (Days)')
plt.ylabel('Complaint Type')
plt.gca().invert_yaxis()  # To display the lowest number at the top
plt.grid()
plt.show()


# In[173]:


weather_data = pd.read_csv('weather_NY_2010_2018Nov.csv')
weather_data['Precipitation'] = weather_data['Percipitation']

weather_data['dt'] = pd.to_datetime(weather_data.Year.astype(str) + '/' + weather_data.Month.astype(str) + '/' + weather_data.Day.astype(str))


# In[174]:


# Calculate yearly average temperatures
yearly_temps = weather_data.groupby('Year').agg({'MeanTemp': 'mean', 'MinTemp': 'mean', 'MaxTemp': 'mean'})

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(yearly_temps.index, yearly_temps['MeanTemp'], label='Mean Temperature', marker='o')
plt.plot(yearly_temps.index, yearly_temps['MinTemp'], label='Minimum Temperature', marker='o')
plt.plot(yearly_temps.index, yearly_temps['MaxTemp'], label='Maximum Temperature', marker='o')

plt.title('Yearly Average Temperatures in New York (2010 - 2018)')
plt.xlabel('Year')
plt.ylabel('Temperature (°F)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# The plot illustrates the yearly average temperatures (mean, minimum, and maximum) in New York from 2010 through 2018. Each line represents the trend for mean, minimum, and maximum temperatures, providing insight into how the climate has varied over these years.

# In[175]:


# Calculate yearly total precipitation
yearly_precipitation = weather_data.groupby('Year')['Percipitation'].sum()

# Plotting the yearly total precipitation
plt.figure(figsize=(12, 6))
plt.bar(yearly_precipitation.index, yearly_precipitation.values, color='skyblue')
plt.title('Yearly Total Precipitation in New York (2010 - 2018)')
plt.xlabel('Year')
plt.ylabel('Precipitation (inches)')
plt.xticks(yearly_precipitation.index)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# The bar chart displays the yearly total precipitation in New York from 2010 through 2018. This visualization helps identify the years with higher or lower amounts of precipitation, indicating variability in rainfall and snowfall over the years.

# In[176]:


# Calculate yearly average wind speed and maximum gust
yearly_wind = weather_data.groupby('Year').agg({'WindSpeed': 'mean', 'Gust': 'max'})

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Average Wind Speed (mph)', color=color)
ax1.plot(yearly_wind.index, yearly_wind['WindSpeed'], label='Average Wind Speed', color=color, marker='o')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Maximum Gust (mph)', color=color)  # we already handled the x-label with ax1
ax2.plot(yearly_wind.index, yearly_wind['Gust'], label='Maximum Gust', color=color, marker='x')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Yearly Wind Patterns in New York (2010 - 2018)')
plt.grid(True)
plt.show()


# The dual-axis line chart illustrates the yearly wind patterns in New York from 2010 through 2018, showing both the average wind speed (in mph) and the maximum gusts recorded each year. The blue line represents the average wind speed, while the red line indicates the maximum gust speeds.
# 
# This visualization provides insight into the general wind conditions and highlights years with particularly strong gusts, reflecting the variability of wind intensity over the observed period.
# 
# Given the analyses conducted so far on the weather dataset, including temperature trends, precipitation patterns, and wind patterns, another insightful exploration could involve examining the relationship between weather conditions and specific weather events, such as days with significant precipitation or high wind speeds, and their occurrence over the years.
# 
# For this next step, let's focus on:
# 
# 1. Days with Significant Precipitation: Identify and visualize the number of days per year with precipitation exceeding a certain threshold, indicating heavy rainfall or significant snowfall events.\
# 2. Days with High Wind Speeds: Analyze and plot the number of days per year with wind speeds exceeding a threshold, highlighting windy conditions or storms.
# 
# #### Days with Significant Precipitation
# Let's define "significant precipitation" as days with precipitation amounts exceeding 0.5 inches, which often indicates heavy rainfall or significant snowfall events. We'll count these days per year to see any trends in their frequency over time.
# 
# 

# In[177]:


# Define a threshold for significant precipitation (in inches)
significant_precipitation_threshold = 0.5

# Filter the dataset for days with precipitation exceeding the threshold
significant_precip_days = weather_data[weather_data['Percipitation'] > significant_precipitation_threshold]

# Count the number of significant precipitation days per year
significant_precip_days_per_year = significant_precip_days.groupby('Year').size()

# Plotting
plt.figure(figsize=(12, 6))
significant_precip_days_per_year.plot(kind='bar', color='navy')
plt.title('Days with Significant Precipitation Per Year (2010 - 2018)')
plt.xlabel('Year')
plt.ylabel('Number of Days')
plt.xticks(rotation=45)
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# The bar chart illustrates the number of days per year with significant precipitation (more than 0.5 inches) in New York from 2010 through 2018. This visualization helps identify years with higher frequencies of heavy rainfall or significant snowfall events, providing insight into variations in extreme weather conditions over the observed period.
# 
# #### Days with High Wind Speeds
# Next, let's define "high wind speeds" as days with wind speeds exceeding 20 mph, which can indicate windy conditions or the presence of storms. We'll count these days per year to observe any trends in their occurrence over time. 

# In[178]:


# Define a threshold for high wind speeds (in mph)
high_wind_speed_threshold = 20

# Filter the dataset for days with wind speeds exceeding the threshold
high_wind_days = weather_data[weather_data['WindSpeed'] > high_wind_speed_threshold]

# Count the number of high wind speed days per year
high_wind_days_per_year = high_wind_days.groupby('Year').size()

# Plotting
plt.figure(figsize=(12, 6))
high_wind_days_per_year.plot(kind='bar', color='darkred')
plt.title('Days with High Wind Speeds Per Year (2010 - 2018)')
plt.xlabel('Year')
plt.ylabel('Number of Days')
plt.xticks(rotation=45)
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# The bar chart showcases the number of days per year with high wind speeds (exceeding 20 mph) in New York from 2010 through 2018. This analysis highlights years with more frequent windy conditions or storms, providing insights into the variability of wind-related weather events over the years.
# 
# These analyses of temperature trends, precipitation patterns, wind patterns, days with significant precipitation, and days with high wind speeds offer a comprehensive overview of the weather conditions in New York over the observed period. 

# In[179]:


requests['Year-Month'] = requests['Created Date'].dt.to_period('M')

# Aggregate 311 requests data to monthly counts
monthly_requests_counts = requests.groupby('Year-Month').size().reset_index(name='Request Count')

# For weather data, ensure we're working with the correct aggregation
weather_data['Year-Month'] = pd.to_datetime(weather_data['Year'].astype(str) + '-' + weather_data['Month'].astype(str)).dt.to_period('M')
monthly_weather_agg = weather_data.groupby('Year-Month').agg({
    'MeanTemp': 'mean',
    'Precipitation': 'sum',  # Total monthly precipitation
    'WindSpeed': 'mean'
}).reset_index()

# Merging the aggregated monthly 311 requests data with the aggregated monthly weather data
combined_data = pd.merge(monthly_requests_counts, monthly_weather_agg, on='Year-Month', how='inner')

combined_data


# In[180]:


import matplotlib.pyplot as plt
import seaborn as sns

# Assuming 'combined_data' is your merged dataset with weather and 311 requests data
fig, axs = plt.subplots(figsize=(12, 6), nrows=1, ncols=2)

sns.scatterplot(x='Precipitation', y='Request Count', data=combined_data, color='b', s=100, ax=axs[0])
requests_precipitation_corr_coef = combined_data['Request Count'].corr(combined_data['Precipitation'], method='spearman')

axs[0].set_title(f'Correlation coefficient={requests_precipitation_corr_coef:.2f}')
axs[0].set_xlabel('Total Precipitation (inches)')
axs[0].set_ylabel('Number of 311 Requests')
axs[0].grid()

sns.scatterplot(x='WindSpeed', y='Request Count', data=combined_data, color='r', s=100, ax=axs[1])
requests_windspeed_corr_coef = combined_data['Request Count'].corr(combined_data['WindSpeed'], method='spearman')

axs[1].set_title(f'Correlation coefficient={requests_windspeed_corr_coef:.2f}')
axs[1].set_xlabel('Wind speed')
axs[1].set_ylabel('Number of 311 Requests')
axs[1].grid()


plt.tight_layout()


# In[181]:


# Filter the 311 dataset for heating complaints
heating_complaints = requests[requests['Complaint Type'].str.contains('Heat', case=False, na=False)]

# Aggregate the count of heating complaints by 'Year-Month'
heating_complaints_monthly = heating_complaints.groupby('Year-Month').size().reset_index(name='Heating Complaints Count')

# Ensure 'Year-Month' is in a plottable format (if necessary, convert to datetime)
heating_complaints_monthly['Year-Month-Datetime'] = heating_complaints_monthly['Year-Month'].dt.to_timestamp()
heating_complaints_monthly

combined_data_with_heating = pd.merge(monthly_weather_agg, heating_complaints_monthly, on='Year-Month', how='inner')

combined_data_with_heating


# In[182]:


fig, ax1 = plt.subplots(figsize=(12,8))

color = 'tab:red'
ax1.set_xlabel('Year-Month')
ax1.set_ylabel('Mean Monthly Temperature', color=color)
ax1.plot(combined_data_with_heating['Year-Month-Datetime'], combined_data_with_heating['MeanTemp'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Heating Complaints Count', color=color)
ax2.plot(combined_data_with_heating['Year-Month-Datetime'], combined_data_with_heating['Heating Complaints Count'], color=color)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Heating Complaints and Mean Monthly Temperature Over Time')
plt.xticks(rotation=45)
fig.tight_layout()
plt.show()


# In[183]:


# Convert 'Created Date' to datetime format, if not already done
requests['Created Date'] = pd.to_datetime(requests['Created Date'])

# Create a new column for the date (without time)
requests['Date'] = requests['Created Date'].dt.date

# Group by the new 'Date' column and count the number of requests per day
daily_requests = requests.groupby('Date').size().reset_index(name='Request Count')

# Assuming 'weather_data' has columns for 'Year', 'Month', and 'Day'
weather_data['Date'] = pd.to_datetime(weather_data[['Year', 'Month', 'Day']]).dt.date

daily_weather_agg = weather_data.groupby('Date').agg({
    'MeanTemp': 'mean',  # Daily average mean temperature
    'MaxTemp': 'mean',  # Daily average maximum temperature
    'MinTemp': 'mean',  # Daily average minimum temperature
    'Precipitation': 'sum',  # Total daily precipitation
    'WindSpeed': 'mean',  # Daily average wind speed
    'SnowDepth': 'max',  # Maximum snow depth for the day
    'Gust': 'max',  # Maximum gust speed for the day
    'Rain':'mean', 
    'MaxSustainedWind': 'mean',
    'SnowDepth':'sum', 
    'SnowIce':'mean'
    # Add any other relevant metrics you wish to include
}).reset_index()

daily_weather_agg['Rain'] = np.round(daily_weather_agg['Rain'])
daily_weather_agg['SnowIce'] = np.round(daily_weather_agg['SnowIce'])

# Merge the daily 311 requests with the daily weather data
combined_data = pd.merge(daily_requests, daily_weather_agg, on='Date', how='inner')

combined_data = combined_data.fillna(0)

combined_data


# In[184]:


combined_data.columns


# In[185]:


# Assuming 'combined_data' includes various weather metrics and types of 311 requests
fig, ax = plt.subplots(figsize=(14,10))

correlation_matrix = np.round(combined_data[['Request Count','MeanTemp', 'MaxTemp', 'MinTemp','Precipitation','WindSpeed', 'SnowDepth', 'Gust', 'Rain',
       'MaxSustainedWind', 'SnowIce',]].corr(),2)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Weather Conditions and 311 Requests')
plt.show()


# In[186]:


from statsmodels.tsa.arima.model import ARIMA

# Assuming 'weather_data' is your historical DataFrame indexed by datetime
last_date = pd.to_datetime(combined_data.Date.max())
forecast_start_date = last_date + pd.Timedelta(days=1)

# Define the forecast period length (e.g., 7 days)
forecast_period = 7

# Create a date range for the forecast period
forecast_dates = pd.date_range(start=forecast_start_date, periods=forecast_period, freq='D')


def forecast_arima(series, order=(1,1,1), steps=7):
    """
    Forecast the next 'steps' points in the series using ARIMA.
    
    Parameters:
    - series: pd.Series, the time series data to forecast.
    - order: tuple, the (p, d, q) order of the ARIMA model.
    - steps: int, the number of steps to forecast ahead.
    
    Returns:
    - forecast: The forecasted values as a pd.Series.
    """
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

weather_metrics = ['MeanTemp', 'MaxTemp', 'MinTemp', 'Percipitation', 'WindSpeed', 'SnowDepth', 'Gust','MaxSustainedWind']
forecast_results = {}

# Assuming 'weather_data' is your DataFrame and it's indexed by a datetime index
for metric in weather_metrics:
    series = weather_data[metric].dropna()  # Drop NA values for simplicity
    forecast = forecast_arima(series, order=(1,1,1), steps=7)  # You may need to adjust the order based on the series
    forecast.index = forecast_dates
    forecast_results[metric] = forecast


forecast_results


# In[187]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Extract day of the week as a feature
combined_data['Date'] = pd.to_datetime(combined_data['Date'])
combined_data['DayOfWeek'] = combined_data['Date'].dt.dayofweek
combined_data['Year'] = combined_data['Date'].dt.year
combined_data['Month'] = combined_data['Date'].dt.month
combined_data['Day'] = combined_data['Date'].dt.day
combined_data['Quarter'] = combined_data['Date'].dt.quarter

combined_data = combined_data.sort_values(by='Date')

# Selecting features and target variable for simplicity
features = ['DayOfWeek','Year','Month','Day','Quarter','MeanTemp','MaxTemp', 'MinTemp', 'Percipitation', 'WindSpeed', 'SnowDepth', 'Gust', 'DayOfWeek','MaxSustainedWind']  # Example features
target = 'Request Count'

encoder = OneHotEncoder()

categorical_features = ['DayOfWeek','Year','Month','Day','Quarter']

# Fit and transform the data
encoded_data = pd.get_dummies(combined_data, columns=categorical_features).drop(columns='Date')

X = encoded_data.drop(columns=target)
y = encoded_data[target]

# Split the data into training and testing sets in chronological order
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

# Initialize and train the linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Predict on the testing set
predictions = lr_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

# Assuming forecast_results is a dictionary containing forecasts for each metric
forecast_data = pd.DataFrame(index=forecast_dates)

for metric, forecast in forecast_results.items():
    forecast_data[metric] = forecast

forecast_data = pd.DataFrame(index=forecast_dates)

for metric, forecast in forecast_results.items():
    forecast_data[metric] = forecast

forecast_data['DayOfWeek'] = forecast_data.index.dayofweek
forecast_data['Year'] = forecast_data.index.year
forecast_data['Month'] = forecast_data.index.month
forecast_data['Day'] = forecast_data.index.day
forecast_data['Quarter'] = forecast_data.index.quarter

test_encoded = pd.get_dummies(forecast_data, columns=categorical_features)

test_encoded_aligned = test_encoded.reindex(columns = X.columns, fill_value=0)

request_count_predictions_lr = lr_model.predict(test_encoded_aligned).astype(int)

forecast_lr = pd.DataFrame(request_count_predictions_lr, index=forecast_dates, columns=['Forecast'])
forecast_lr


# In[188]:


from sklearn.ensemble import RandomForestRegressor

# Initialize the RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth = 10)  # You can adjust these parameters

# Train the model
rf_model.fit(X_train, y_train)

# Predict on the testing set
predictions = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

request_count_predictions_rf = rf_model.predict(test_encoded_aligned).astype(int)
forecast_rf = pd.DataFrame(request_count_predictions_rf, index=forecast_dates, columns=['Forecast'])
forecast_rf


# In[189]:


import xgboost as xgb

# Initialize the XGBRegressor
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', 
                             colsample_bytree = 0.3, 
                             learning_rate = 0.1,
                             max_depth = 5, 
                             alpha = 10, 
                             n_estimators = 200)

# Train the model
xgb_model.fit(X_train, y_train)

# Predict on the testing set
predictions = xgb_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")

request_count_predictions_xgb = xgb_model.predict(test_encoded_aligned).astype(int)
forecast_xgb = pd.DataFrame(request_count_predictions_xgb, index=forecast_dates, columns=['Forecast'])
forecast_xgb


# In[190]:


fig, ax = plt.subplots(figsize=(12,8))

sns.lineplot(data=daily_requests.loc[(pd.to_datetime(daily_requests['Date']) > pd.to_datetime('2018-10-15')) & 
                                     (pd.to_datetime(daily_requests['Date']) <= pd.to_datetime(test_encoded_aligned.index[-1]))], 
                                     x='Date', y='Request Count', color='k', label='original', lw=4, linestyle='--')
sns.lineplot(data=daily_requests.loc[(pd.to_datetime(daily_requests['Date']) > pd.to_datetime('2018-11-12')) & 
                                     (pd.to_datetime(daily_requests['Date']) <= pd.to_datetime(test_encoded_aligned.index[-1]))], 
                                     x='Date', y='Request Count', color='r', label='original', lw=4, linestyle='--')

sns.lineplot(x=forecast_lr.index, y=forecast_lr['Forecast'], label='LR', lw=4)
sns.lineplot(x=forecast_rf.index, y=forecast_rf['Forecast'],  label='RF', lw=4)
sns.lineplot(x=forecast_xgb.index, y=forecast_xgb['Forecast'], label='XGB', lw=4)
plt.setp(ax.get_xticklabels(), rotation=90, ha='center')

ax.legend()

ax.grid()


# In[ ]:




