import pandas as pd

county_data = pd.read_csv('/Users/briankalinowski/PycharmProjects/CIS600/zecon/County_time_series.csv')
metro_data = pd.read_csv('/Users/briankalinowski/PycharmProjects/CIS600/zecon/Metro_time_series.csv')
neighborhood_data = pd.read_csv('/Users/briankalinowski/PycharmProjects/CIS600/zecon/Neighborhood_time_series.csv')
state_data = pd.read_csv('/Users/briankalinowski/PycharmProjects/CIS600/zecon/State_time_series.csv')
zip_data = pd.read_csv('/Users/briankalinowski/PycharmProjects/CIS600/zecon/Zip_time_series.csv')

# date time column
county_data.Date = pd.to_datetime(county_data.Date)
metro_data.Date = pd.to_datetime(metro_data.Date)
neighborhood_data.Date = pd.to_datetime(neighborhood_data.Date)
state_data.Date = pd.to_datetime(state_data.Date)
zip_data.Date = pd.to_datetime(zip_data.Date)

# drop na if NAN in all columns
county_data.dropna(how='all', inplace=True)
metro_data.dropna(how='all', inplace=True)
neighborhood_data.dropna(how='all', inplace=True)
state_data.dropna(how='all', inplace=True)
zip_data.dropna(how='all', inplace=True)

# write to file
county_data.to_csv('/Users/briankalinowski/Desktop/CIS600_DataMining/Zillow_Data_Clean/county_clean.csv', index=None, header=True)
metro_data.to_csv('/Users/briankalinowski/Desktop/CIS600_DataMining/Zillow_Data_Clean/metro_clean.csv', index=None, header=True)
neighborhood_data.to_csv('/Users/briankalinowski/Desktop/CIS600_DataMining/Zillow_Data_Clean/neighborhood_clean.csv', index=None, header=True)
state_data.to_csv('/Users/briankalinowski/Desktop/CIS600_DataMining/Zillow_Data_Clean/state_clean.csv', index=None, header=True)
zip_data.to_csv('/Users/briankalinowski/Desktop/CIS600_DataMining/Zillow_Data_Clean/zip_clean.csv', index=None, header=True)

