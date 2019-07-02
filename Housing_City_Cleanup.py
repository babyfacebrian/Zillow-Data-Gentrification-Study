import pandas as pd
import ZillowProject.City_Preprocessing as zillowPro

# Geo and housing features
housing_features = ['Date', 'RegionName', 'PriceToRentRatio_AllHomes',
                    'ZHVI_AllHomes', 'ZHVIPerSqft_AllHomes',
                    'ZHVI_BottomTier', 'ZHVI_MiddleTier', 'ZHVI_TopTier']

# all our housing measurement features
housing_values = ['PriceToRentRatio_AllHomes', 'ZHVI_AllHomes',
                  'ZHVIPerSqft_AllHomes', 'ZHVI_BottomTier', 'ZHVI_MiddleTier', 'ZHVI_TopTier']

lookup_ix = ['State', 'City']
full_ix = ['State', 'City', 'Date']

city_raw = pd.read_csv('/Users/briankalinowski/PycharmProjects/CIS600/zecon/City_time_series.csv')

city_data_clean = zillowPro.clean_city_data(city_raw, housing_features, housing_values)
city_data_clean = zillowPro.merge_location_data(city_data_clean)

city_data_clean.to_csv('/Users/briankalinowski/Desktop/CIS600_DataMining/Zillow_Data_Clean/city_housing_data.csv',
                       index=None, header=True)

'''
h = ['Date', 'RegionName', 'InventorySeasonallyAdjusted_AllHomes', 'InventoryRaw_AllHomes',
     'MedianListingPricePerSqft_1Bedroom', 'MedianListingPricePerSqft_2Bedroom', 'MedianListingPricePerSqft_3Bedroom',
     'MedianListingPricePerSqft_4Bedroom', 'MedianListingPricePerSqft_5BedroomOrMore',
     'MedianListingPricePerSqft_AllHomes',
     'MedianListingPricePerSqft_CondoCoop', 'MedianListingPricePerSqft_DuplexTriplex',
     'MedianListingPricePerSqft_SingleFamilyResidence', 'MedianListingPrice_1Bedroom', 'MedianListingPrice_2Bedroom',
     'MedianListingPrice_3Bedroom', 'MedianListingPrice_4Bedroom', 'MedianListingPrice_5BedroomOrMore',
     'MedianListingPrice_AllHomes', 'MedianListingPrice_CondoCoop', 'MedianListingPrice_DuplexTriplex',
     'MedianListingPrice_SingleFamilyResidence', 'MedianPctOfPriceReduction_AllHomes',
     'MedianPctOfPriceReduction_CondoCoop', 'MedianPctOfPriceReduction_SingleFamilyResidence',
     'MedianPriceCutDollar_AllHomes', 'MedianPriceCutDollar_CondoCoop', 'MedianPriceCutDollar_SingleFamilyResidence',
     'MedianRentalPricePerSqft_1Bedroom', 'MedianRentalPricePerSqft_2Bedroom', 'MedianRentalPricePerSqft_3Bedroom',
     'MedianRentalPricePerSqft_4Bedroom', 'MedianRentalPricePerSqft_5BedroomOrMore',
     'MedianRentalPricePerSqft_AllHomes',
     'MedianRentalPricePerSqft_CondoCoop', 'MedianRentalPricePerSqft_DuplexTriplex',
     'MedianRentalPricePerSqft_MultiFamilyResidence5PlusUnits', 'MedianRentalPricePerSqft_SingleFamilyResidence',
     'MedianRentalPricePerSqft_Studio', 'MedianRentalPrice_1Bedroom', 'MedianRentalPrice_2Bedroom',
     'MedianRentalPrice_3Bedroom', 'MedianRentalPrice_4Bedroom', 'MedianRentalPrice_5BedroomOrMore',
     'MedianRentalPrice_AllHomes', 'MedianRentalPrice_CondoCoop', 'MedianRentalPrice_DuplexTriplex',
     'MedianRentalPrice_MultiFamilyResidence5PlusUnits', 'MedianRentalPrice_SingleFamilyResidence',
     'MedianRentalPrice_Studio', 'ZHVIPerSqft_AllHomes', 'PctOfHomesDecreasingInValues_AllHomes',
     'PctOfHomesIncreasingInValues_AllHomes', 'PctOfHomesSellingForGain_AllHomes', 'PctOfHomesSellingForLoss_AllHomes',
     'PctOfListingsWithPriceReductionsSeasAdj_AllHomes', 'PctOfListingsWithPriceReductionsSeasAdj_CondoCoop',
     'PctOfListingsWithPriceReductionsSeasAdj_SingleFamilyResidence', 'PctOfListingsWithPriceReductions_AllHomes',
     'PctOfListingsWithPriceReductions_CondoCoop', 'PctOfListingsWithPriceReductions_SingleFamilyResidence',
     'PriceToRentRatio_AllHomes', 'Sale_Counts', 'Sale_Counts_Seas_Adj', 'Sale_Prices', 'ZHVI_1bedroom',
     'ZHVI_2bedroom',
     'ZHVI_3bedroom', 'ZHVI_4bedroom', 'ZHVI_5BedroomOrMore', 'ZHVI_AllHomes', 'ZHVI_BottomTier', 'ZHVI_CondoCoop',
     'ZHVI_MiddleTier', 'ZHVI_SingleFamilyResidence', 'ZHVI_TopTier', 'ZRI_AllHomes', 'ZRI_AllHomesPlusMultifamily',
     'ZriPerSqft_AllHomes', 'Zri_MultiFamilyResidenceRental', 'Zri_SingleFamilyResidenceRental']
'''
