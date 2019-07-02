import pandas as pd
import ZillowProject.City_Preprocessing as zillowPro

# Geo and Rental features
rental_features = ['Date', 'RegionName',
                   'PriceToRentRatio_AllHomes',
                   'ZRI_AllHomes',
                   'ZriPerSqft_AllHomes',
                   'Zri_SingleFamilyResidenceRental']

# all our rental measurement features
rental_values = ['PriceToRentRatio_AllHomes',
                 'ZRI_AllHomes',
                 'ZriPerSqft_AllHomes',
                 'Zri_SingleFamilyResidenceRental']

lookup_ix = ['State', 'City']
full_ix = ['State', 'City', 'Date']

city_raw = pd.read_csv('/Users/briankalinowski/PycharmProjects/CIS600/zecon/City_time_series.csv')

city_data_clean = zillowPro.clean_city_data(city_raw, rental_features, rental_values)
city_data_clean = zillowPro.merge_location_data(city_data_clean)

city_data_clean.to_csv('/Users/briankalinowski/Desktop/CIS600_DataMining/Zillow_Data_Clean/city_rental_data_.csv',
                       index=None, header=True)


