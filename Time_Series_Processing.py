import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans


def clean_zillow_ts_data(zillow_ts_data: pd.DataFrame, lookup: list) -> pd.DataFrame:
    # sorting
    zillow_ts_data.set_index(lookup, inplace=True)
    zillow_ts_data.sort_index(inplace=True)
    zillow_ts_data.reset_index(inplace=True)

    # set all metric columns to float
    metrics = [col for col in list(zillow_ts_data.columns) if col not in lookup]
    zillow_ts_data[metrics] = zillow_ts_data[metrics].astype(float)

    # back-fill NaN values by ROW
    zillow_ts_data[metrics] = zillow_ts_data[metrics].fillna(axis=1, method='bfill')





