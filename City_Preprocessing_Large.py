import pandas as pd


def clean_city_data(data: pd.DataFrame, features: list, search_values: list) -> pd.DataFrame:
    """
    :return: Filters city data with features columns, Cleans NaN values from search_values
    """
    data = data.filter(features)
    data.dropna(how='all', subset=search_values, inplace=True)

    # convert date column to pd Datetime
    if 'Date' in list(data.columns):
        data.Date = pd.to_datetime(data.Date)
    return data


def merge_location_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    :return: Merge proper city/state names to city data
    """
    # ADD YOUR LOCAL PATH FOR cities_crosswalk.csv
    city_locations = pd.read_csv('/Users/briankalinowski/PycharmProjects/CIS600/zecon/cities_crosswalk.csv')
    city_locations.rename(columns={'Unique_City_ID': 'RegionName'}, inplace=True)
    data = data.merge(city_locations, on='RegionName', how='left')
    data = data.drop(['RegionName', 'County'], axis=1)
    return data


def get_city_state_lookup(data: pd.DataFrame, indices: list) -> list:
    """
    :return: List of tuples of unique indices
    """
    lookup = data.filter(indices).dropna()
    lookup.set_index(indices, inplace=True)
    lookup.sort_index(inplace=True)
    index_set = list(set(lookup.index.tolist()))
    return index_set


def set_city_indices(data: pd.DataFrame, indices: list) -> pd.DataFrame:
    data.set_index(indices, inplace=True)
    data.sort_index(inplace=True)
    return data


def get_city_feature_aggregation(data: pd.DataFrame, indices: list) -> pd.DataFrame:
    """
    :return: Aggregates the features by year for each level of the index
    """
    data_aggregate = pd.DataFrame()

    for i in range(0, len(indices)):
        single_city_df: pd.DataFrame = data.xs([indices[i][0], indices[i][1]], level=['State', 'City'])
        single_city_df.fillna(method='bfill', inplace=True)

        if single_city_df.isna().values.any() == False:
            single_city_df = single_city_df.add_prefix('AVG_')
            single_city_df.reset_index(inplace=True)
            single_city_df.insert(0, "State", indices[i][0])
            single_city_df.insert(1, 'City', indices[i][1])
            data_aggregate = data_aggregate.append(single_city_df, ignore_index=True)
    return data_aggregate


def process_city_data(data, features, values, lookup_indices, final_indices) -> pd.DataFrame:
    data = clean_city_data(data, features, values)
    data = merge_location_data(data)
    lookup = get_city_state_lookup(data, lookup_indices)
    data = set_city_indices(data, final_indices)
    data = get_city_feature_aggregation(data, lookup)
    return data
