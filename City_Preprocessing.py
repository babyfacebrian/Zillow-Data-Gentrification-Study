import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def clean_city_data(data: pd.DataFrame, features: list, search_values: list) -> pd.DataFrame:
    """
    :return: Filters city data with features columns, Cleans NaN values from search_values
    """
    city_data = data.filter(features)
    city_data.dropna(how='all', subset=search_values, inplace=True)

    # convert date column to pd Datetime
    if 'Date' in list(city_data.columns):
        city_data.Date = pd.to_datetime(city_data.Date)
    return city_data


def merge_location_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    :return: Merge proper city/state names to city data
    """
    # ADD YOUR LOCAL PATH FOR cities_crosswalk.csv
    city_locations = pd.read_csv('/Users/briankalinowski/PycharmProjects/CIS600/zecon/cities_crosswalk.csv')
    city_locations.rename(columns={'Unique_City_ID': 'RegionName'}, inplace=True)
    city_data = data.merge(city_locations, on='RegionName', how='left')
    city_data = city_data.drop(['RegionName', 'County'], axis=1)
    return city_data


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
            single_city_df = single_city_df.resample('A').mean()
            single_city_df = single_city_df.add_prefix('AVG_')
            single_city_df.set_index(single_city_df.index.year, inplace=True)
            single_city_df.reset_index(inplace=True)
            single_city_df.insert(0, "State", indices[i][0])
            single_city_df.insert(1, 'City', indices[i][1])
            data_aggregate = data_aggregate.append(single_city_df, ignore_index=True)
    return data_aggregate


def get_gentrification_control(data: pd.DataFrame) -> pd.DataFrame:
    """
    :return: Aggregates the known gentrified cities and returns DataFrame of yearly avg
    """
    gent = [('NC', 'Asheville'), ('TN', 'Nashville'), ('CA', 'Oakland'), ('SC', 'Charleston'), ('CA', 'Anaheim'),
            ('CA', 'Berkeley'), ('WA', 'Seattle'), ('TX', 'Austin'), ('CA', 'Los Angeles'), ('CA', 'San Diego'),
            ('TX', 'Midland'), ('DC', 'Washington'), ('OR', 'Portland'), ('CA', 'Sacramento'), ('NY', 'New York'),
            ('MI', 'Royal Oak'), ('AR', 'Bentonville'), ('CA', 'Costa Mesa'), ('CA', 'San Marcos'), ('MI', 'Ann Arbor'),
            ('NJ', 'Jersey City'), ('MA', 'Somerville'), ('CO', 'Thornton'), ('CA', 'Vista'), ('CA', 'Long Beach'),
            ('PA', 'Pittsburgh'), ('MA', 'Quincy'), ('CA', 'Napa'), ('OR', 'Hillsboro'), ('CO', 'Denver'),
            ('CA', 'Hayward')]

    control_aggregate = pd.DataFrame()

    for i in range(0, len(gent)):
        control_df: pd.DataFrame = data.xs([gent[i][0], gent[i][1]], level=['State', 'City'])
        control_df.fillna(method='bfill', inplace=True)
        if control_df.isna().values.any() == False:
            control_aggregate = control_aggregate.append(control_df, ignore_index=False)  # False!!!!

    control_aggregate = control_aggregate.resample('A').mean()
    control_aggregate = control_aggregate.add_prefix('AVG_')
    control_aggregate.set_index(control_aggregate.index.year, inplace=True)
    control_aggregate.reset_index(inplace=True)
    control_aggregate.insert(0, "State", 'GENT_CONTROL')
    control_aggregate.insert(1, 'City', 'GENT_CONTROL')
    return control_aggregate


def min_max_scale(data: pd.DataFrame, features: list) -> pd.DataFrame:
    min_max = MinMaxScaler()
    data[features] = pd.DataFrame(min_max.fit_transform(data[features]))
    return data


def process_city_data(data, features, values, lookup_indices, final_indices) -> pd.DataFrame:
    city_data = clean_city_data(data, features, values)
    city_data = merge_location_data(city_data)
    lookup = get_city_state_lookup(city_data, lookup_indices)
    city_data = set_city_indices(city_data, final_indices)
    control = get_gentrification_control(city_data)
    city_data = get_city_feature_aggregation(city_data, lookup)
    city_data = city_data.append(control, ignore_index=True)
    return city_data
