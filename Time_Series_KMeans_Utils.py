import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesResampler


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

    # min-max scale metrics
    min_max = MinMaxScaler()
    zillow_ts_data[metrics] = pd.DataFrame(min_max.fit_transform(zillow_ts_data[metrics]))
    return zillow_ts_data


def run_time_series_kmeans(ts_data: pd.DataFrame, labels: list, sample: int) -> pd.DataFrame:
    # drop our geo cols
    features_df = ts_data.drop(columns=labels)
    geo_labels_df = ts_data.filter(labels)

    # tslearn TimeSeriesKMeans cluster
    ts_km = TimeSeriesKMeans(n_clusters=5, n_init=1, metric='dtw', max_iter=5, max_iter_barycenter=5, dtw_inertia=True)

    # re-sample by year
    features_resampled = TimeSeriesResampler(sz=sample).fit_transform(features_df.values)
    dtw_predict = ts_km.fit_predict(features_resampled)

    # DTW predictions df
    dtw_predict_df = pd.DataFrame(dtw_predict, columns=['dtw_cluster_prediction'])

    # merge cluster prediction to geo data
    geo_clusters_df = geo_labels_df.merge(dtw_predict_df, how='outer', left_index=True, right_index=True)
    geo_clusters_df = geo_clusters_df.astype({'dtw_cluster_prediction': float})
    return geo_clusters_df



