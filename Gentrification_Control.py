import pandas as pd
import numpy as np
from tslearn.metrics import dtw
from sklearn.preprocessing import KBinsDiscretizer


def get_city_state_lookup(data: pd.DataFrame, indices: list) -> list:
    """
    :return: List of tuples of unique indices
    """
    lookup = data.filter(indices).dropna()
    lookup.set_index(indices, inplace=True)
    lookup.sort_index(inplace=True)
    index_set = list(set(lookup.index.tolist()))
    return index_set


def create_dtw_scores(data_yearly_agg: pd.DataFrame) -> pd.DataFrame:
    # remove our GENT_CONTROL city/state from our city data
    state_city_lookup = get_city_state_lookup(data_yearly_agg, ['State', 'City'])
    state_city_lookup.remove(('GENT_CONTROL', 'GENT_CONTROL'))

    # dtw df for each city when compared to our GENT_CONTROL average
    dtw_df = pd.DataFrame(columns=['State', 'City', 'dtw_score'])

    # Gent. control df
    control_df = data_yearly_agg.loc[data_yearly_agg.State == 'GENT_CONTROL'].loc[data_yearly_agg.City == 'GENT_CONTROL']
    control_df.drop(columns=['State', 'City', 'Date'], inplace=True)

    for i in range(0, len(state_city_lookup)):
        # State/City names
        state = state_city_lookup[i][0]
        city = state_city_lookup[i][1]

        # locate each city and drop columns
        single_city_df = data_yearly_agg.loc[data_yearly_agg.State == state].loc[data_yearly_agg.City == city]
        single_city_df.drop(columns=['State', 'City', 'Date'], inplace=True)

        # calculate DTW score
        dtw_score = dtw(single_city_df.values, control_df.values)
        # Append city row to dtw df
        dtw_df = dtw_df.append({'State': state, 'City': city, 'dtw_score': dtw_score}, ignore_index=True)

    # add our GENT_CONTROL back with a 0.0 DTW score
    dtw_df = dtw_df.append({'State': 'GENT_CONTROL', 'City': 'GENT_CONTROL', 'dtw_score': 0.0}, ignore_index=True)
    return dtw_df


def encode_dtw_labels(dtw_data: pd.DataFrame) -> pd.DataFrame:
    # Encoder to encode the range of DTW scores (5 buckets) into a gentrification label
    encoder = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='kmeans')
    dtw_arr = np.reshape(dtw_data.dtw_score.values, (-1, 1))
    encoded_df = encoder.fit_transform(dtw_arr)
    # New df with labels and scores
    encoded_df = pd.DataFrame(encoded_df, columns=['dtw_value'])
    # 5 gentrification categories
    dtw_values_dict = {0.0: 'HIGH_GENT', 1.0: 'MID_GENT', 2.0: 'NORMAL_GENT', 3.0: 'LOW_GENT', 4.0: 'NO_GENT'}
    # map dtw scores to labels
    encoded_df['dtw_label'] = encoded_df.dtw_value.map(dtw_values_dict)

    # merge to data with geo columns
    dtw_labeled = dtw_data.merge(encoded_df, how='outer', left_index=True, right_index=True)
    return dtw_labeled


def set_geo_dtw_data(geo_data: pd.DataFrame) -> pd.DataFrame:
    # calculate dtw scores for all geo rows
    dtw_df = create_dtw_scores(geo_data)
    # encode and label dtw scores
    dtw_labeled = encode_dtw_labels(dtw_df)
    # reset/reorder by state/city
    dtw_labeled.set_index(['State', 'City'], inplace=True)
    dtw_labeled.sort_index(inplace=True)
    dtw_labeled.reset_index(inplace=True)
    return dtw_labeled




