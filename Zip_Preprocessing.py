import pandas as pd


def clean_zip_data(data: pd.DataFrame, features: list, search_values: list) -> pd.DataFrame:
    """
    :return: Filters zip data with features columns, Cleans NaN values from search_values
    """
    data = data.filter(features)
    data.dropna(how='all', subset=search_values, inplace=True)
    # convert date column to pd Datetime
    if 'Date' in list(data.columns):
        data.Date = pd.to_datetime(data.Date)
    return data


def merge_location_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    :return: Merge proper city/state names to zip data
    """
    # ADD YOUR LOCAL PATH FOR cities_crosswalk.csv
    zipcode_map = pd.read_csv('/Users/briankalinowski/Desktop/CIS600_DataMining/Zillow_Data_Clean/Zillow_zipcode_map.csv')
    data.rename(columns={'RegionName': 'ZIP'}, inplace=True)
    data = data.merge(zipcode_map, on='ZIP', how='left')
    return data


def get_city_state_zip_lookup(data: pd.DataFrame, indices: list) -> list:
    """
    :return: List of tuples of unique indices
    """
    lookup = data.filter(indices).dropna()
    lookup.set_index(indices, inplace=True)
    lookup.sort_index(inplace=True)
    index_set = list(set(lookup.index.tolist()))
    return index_set


def set_geo_indices(data: pd.DataFrame, indices: list) -> pd.DataFrame:
    data.set_index(indices, inplace=True)
    data.sort_index(inplace=True)
    return data


def get_zip_feature_aggregation(data: pd.DataFrame, indices: list) -> pd.DataFrame:
    """
    :return: Aggregates the features by year for each level of the index
    """
    data_aggregate = pd.DataFrame()

    for i in range(0, len(indices)):
        single_zip_df: pd.DataFrame = data.xs([indices[i][0], indices[i][1], indices[i][2]], level=['State', 'City', 'ZIP'])
        single_zip_df.fillna(method='bfill', inplace=True)

        if single_zip_df.isna().values.any() == False:
            single_zip_df = single_zip_df.resample('A').mean()
            single_zip_df = single_zip_df.add_prefix('AVG_')
            single_zip_df.set_index(single_zip_df.index.year, inplace=True)
            single_zip_df.reset_index(inplace=True)
            single_zip_df.insert(0, "State", indices[i][0])
            single_zip_df.insert(1, 'City', indices[i][1])
            single_zip_df.insert(2, 'ZIP', indices[i][2])
            data_aggregate = data_aggregate.append(single_zip_df, ignore_index=True)

    return data_aggregate


def process_zip_data(data, features, values, lookup_indices, final_indices) -> pd.DataFrame:
    data = clean_zip_data(data, features, values)
    data = merge_location_data(data)
    lookup = get_city_state_zip_lookup(data, lookup_indices)
    data = set_geo_indices(data, final_indices)
    data = get_zip_feature_aggregation(data, lookup)
    return data
