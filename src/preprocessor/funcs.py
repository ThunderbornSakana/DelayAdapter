import numpy as np
import pandas as pd
from collections import defaultdict

NUM_SEC_PER_MINUTE = 60
NUM_SEC_PER_HOUR = NUM_SEC_PER_MINUTE * 60
NUM_SEC_PER_DAY = NUM_SEC_PER_HOUR * 24

def create_empty_array():
    return np.array([])


# Each unique value will be encoded as starting from 0, 1, 2, .....
def get_ohe(uni_vals):
    ohedict = defaultdict(create_empty_array)
    i = 0
    for val in uni_vals:
        ohe_vec = i
        ohedict[val] = ohe_vec
        i += 1
    return ohedict


def taobao_filter_duplicates(group):
    # Filter out all rows where timeStamp is less than or equal to timeStamp in preview.
    filtered_group = group[group['timeStamp'] > group['timeStamp_pv']]
    if not filtered_group.empty:
        # Return the row with the smallest timeStamp.
        return filtered_group.nsmallest(1, 'timeStamp')
    return pd.DataFrame()


def train_test_split(data, train_start, train_end, num_test, timestamp_col):
    train_df = data[(data[timestamp_col] >= train_start * NUM_SEC_PER_DAY) & (
        data[timestamp_col] < (train_end + 1) * NUM_SEC_PER_DAY)]
    test_df = data[(data[timestamp_col] >= (train_end + 1) * NUM_SEC_PER_DAY) & (
        data[timestamp_col] < (train_end + 1 + num_test) * NUM_SEC_PER_DAY)]
    train_y = train_df['label']
    train_x = train_df.drop('label', axis=1)
    test_y = test_df['label']
    test_x = test_df.drop('label', axis=1)
    return train_x, train_y, test_x, test_y


def data_analytics(data, timestamp_col):
    print("Total duration of dataset: " + str((
        data[timestamp_col].max() - data[timestamp_col].min()) / NUM_SEC_PER_DAY
        )+ " days")
    print("Total number of clicks: " + str(len(data)))
    num_conversions = len(data[data['label'] == 1])
    print("Total number of conversions: " + str(num_conversions))
    print("Conversion rate: " + str(num_conversions / (len(data) + num_conversions)))
    print("Average delay in days: " + str(data[data['delay_in_days'] >= 0][
        'delay_in_days'].mean()))
    qs = np.percentile(data[data['delay_in_days'] >= 0]['delay_in_days'], [
        25, 50, 75, 90, 95, 99.9, 100])
    print("25, 50, 75, 90, 95, 99.9, 100 percentiles:", qs)
    return


def preprocess_criteo(train_start, train_end, num_test):
    # Load raw data and create label column.
    all_cols = ["timestamp", "convertTimestamp", "i1", "i2", "i3", "i4", 
                "i5", "i6", "i7", "i8", "c1", "c2", "c3", "c4", "c5", 
                "c6", "c7", "c8", "c9"]
    timestamp_col = "timestamp"
    raw = pd.read_csv("../../data/criteo/data.txt", sep="\t", header=None, 
                      names=all_cols)
    raw['label'] = raw['convertTimestamp'].notnull().astype('int')
    # Get the time difference between conversions and clicks.
    raw['delay'] = np.where(raw['label'] == 1, 
                            raw['convertTimestamp'] - raw[timestamp_col], -1)
    raw['delay_in_days'] = raw['delay'] / NUM_SEC_PER_DAY
    # Make sure all timestamps start from 0.
    min_timestamp = raw[timestamp_col].min()
    raw[timestamp_col] -= min_timestamp
    raw['convertTimestamp'] -= min_timestamp
    # For integer value columns, replace NaNs with mean values.
    for i in range(1, 9):
        col = 'i' + str(i)
        mean_value = raw[col].mean()
        raw[col].fillna(value=mean_value, inplace=True)
    # For categorical columns, replace NaNs with "nan_flag."
    for i in range(1, 10):
        col = 'c' + str(i)
        raw[col].fillna(value="nan_flag", inplace=True)
    # Apply min-max normalization for every integer value column.
    min_max = {'i' + str(i): [raw['i' + str(i)].min(), raw['i' + str(i)].max()] 
               for i in range(1, 9)}
    for i in range(1, 9):
        col = 'i' + str(i)
        min_val, max_val = min_max[col]
        raw[col] = raw[col].apply(lambda x: (x - min_val) / (max_val - min_val))
    # Get the unique values for all categorical columns.
    cat_dict = defaultdict(list)
    for i in range(1, 10):
        col = 'c' + str(i)
        unique_val = list(raw[col].unique())
        cat_dict[col] = unique_val
    # Apply encoding for all categorical columns.
    all_cat = defaultdict(create_empty_array)
    for i in range(1, 10):
        col = 'c' + str(i)
        all_cat[col] = get_ohe(cat_dict[col])
    for i in range(1, 10):
        col = 'c' + str(i)
        val_dict = all_cat[col]
        raw[col] = raw[col].apply(lambda x: val_dict[x])
    # Get the statistics of the dataset.
    data_analytics(raw, timestamp_col)
    # Get the number of unique values for all categorical features into a list.
    cat_fields = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    total_cat = raw[cat_fields]
    cat_dims = [total_cat[col].unique().size for col in cat_fields]
    # Split the dataset into train and test.
    train_x, train_y, test_x, test_y = train_test_split(raw, train_start, 
                                                        train_end, num_test, 
                                                        timestamp_col)
    return train_x, train_y, test_x, test_y, cat_dims


def preprocess_tencent(train_start, train_end, num_test):
    # Load raw data.
    data_dir = "../../data/tencent"
    train = pd.read_csv(f"{data_dir}/train.csv")
    user = pd.read_csv(f"{data_dir}/user.csv")
    position = pd.read_csv(f"{data_dir}/position.csv")
    app_cat = pd.read_csv(f"{data_dir}/app_categories.csv")
    ad = pd.read_csv(f"{data_dir}/ad.csv")
    ad = ad.merge(app_cat, on='appID', how='left')
    # Convert timestamps to datetime format.
    timestamp_col = "clickTime"
    train[timestamp_col] = pd.to_datetime(train[timestamp_col], format='%d%H%M%S')
    train['conversionTime'] = pd.to_datetime(train['conversionTime'], 
                                             format='%d%H%M%S')
    # Abandon the data in the last two days.
    raw = train[train[timestamp_col] < pd.Timestamp('1900-01-29')]
    # Merge all dataframes.
    raw = raw.merge(user, on='userID', how='left')
    raw = raw.merge(position, on='positionID', how='left')
    raw = raw.merge(ad, on='creativeID', how='left')
    # Remove some rows based on certain conditions on selected columns.
    advertiserID_2_rmv = raw.advertiserID.value_counts().sort_values().head(28).index.tolist()
    residence_2_rmv = raw.residence.value_counts().sort_values().head(14).index.tolist()
    appID_2_rmv = raw.appID.value_counts().sort_values().head(14).index.tolist()
    raw = raw[~raw['advertiserID'].isin(advertiserID_2_rmv)]
    raw = raw[~raw['residence'].isin(residence_2_rmv)]
    raw = raw[~raw['appID'].isin(appID_2_rmv)]
    # Remove selected columns.
    cols_2_rmv = ["creativeID", "userID", "positionID", "adID", "camgaignID"]
    raw.drop(cols_2_rmv, axis=1, inplace=True)
    # Manipulate timestamps.
    earliest = raw[timestamp_col].min()
    raw['clickTime_sec'] = (raw[timestamp_col] - earliest).dt.total_seconds()
    raw['conversionTime_sec'] = np.where(raw['label'] == 1, (
        raw['conversionTime'] - earliest).dt.total_seconds(), -1)
    raw.drop(timestamp_col, axis=1, inplace=True)
    raw.drop('conversionTime', axis=1, inplace=True)
    # Get the time difference between conversions and clicks.
    timestamp_col = 'clickTime_sec'
    raw['delay'] = np.where(raw['label'] == 1, 
                            raw['conversionTime_sec'] - raw[timestamp_col], -1)
    raw['delay_in_days'] = raw['delay'] / NUM_SEC_PER_DAY
    # Get the statistics of the dataset.
    data_analytics(raw, timestamp_col)
    # Get the unique values for all categorical columns.
    cat_feats = ["connectionType", "telecomsOperator", "age", "gender", 
                 "education", "marriageStatus", "haveBaby", "hometown", 
                 "sitesetID", "positionType", "appID", "appPlatform", 
                 "appCategory", "residence", "advertiserID"]
    cat_dict = defaultdict(list)
    for col in cat_feats:
        unique_val = list(raw[col].unique())
        cat_dict[col] = unique_val
    # Apply encoding for all categorical columns.
    all_cat = defaultdict(create_empty_array)
    for col in cat_feats:
        all_cat[col] = get_ohe(cat_dict[col])
    for col in cat_feats:
        val_dict = all_cat[col]
        raw[col] = raw[col].apply(lambda x: val_dict[x])
    # Get the number of unique values for all categorical features into a list.
    total_cat = raw[cat_feats]
    cat_dims = [total_cat[col].unique().size for col in cat_feats]
    # Split the dataset into train and test.
    train_x, train_y, test_x, test_y = train_test_split(raw, train_start, 
                                                        train_end, num_test, 
                                                        timestamp_col)
    return train_x, train_y, test_x, test_y, cat_dims


def preprocess_taobao(train_start, train_end, num_test, from_midpoint=True):
    data_dir = "../../data/taobao"
    timestamp_col = 'pvTime'
    cat_feats = ['UserID', 'ProductID', 'Product_type_ID']
    if from_midpoint:
        raw = pd.read_csv(f"{data_dir}/preprocessed_taobao.csv")
    else:
        # Load raw data and parse dates.
        all_cols = ['UserID', 'ProductID', 'Product_type_ID', 'Behavior', 'timeStamp']
        raw = pd.read_csv(f"{data_dir}/UserBehavior.csv", names=all_cols)
        raw['timeStamp'] = pd.to_datetime(raw['timeStamp'], unit='s')
        # Split into preview and buy.
        pv = raw[raw['Behavior'] == 'pv']
        buy = raw[raw['Behavior'] == 'buy']
        # Remove data points with abnormal timestamps.
        start_date = pd.to_datetime('2017-11-25')
        end_date = pd.to_datetime('2017-12-03')
        pv = pv[(pv['timeStamp'] >= start_date) & (pv['timeStamp'] <= end_date)]
        buy = buy[buy['timeStamp'] >= start_date]
        # Only keep the earliest pv records.
        pv = pv.sort_values(by='timeStamp'
                            ).drop_duplicates(subset=[
                                "UserID", "ProductID", "Product_type_ID"], keep="first")
        # Filter out non-matching rows in buy.
        buy = buy.merge(pv[['UserID', 'ProductID', 'Product_type_ID']], on=[
            'UserID', 'ProductID', 'Product_type_ID'])
        # Further filter buy based on timeStamp criteria.
        buy = buy.merge(pv[['UserID', 'ProductID', 
                            'Product_type_ID', 'timeStamp']], on=[
                                'UserID', 'ProductID', 'Product_type_ID'], 
                                how='left', suffixes=('', '_pv'))
        buy = buy.groupby(['UserID', 'ProductID', 'Product_type_ID'
                           ]).apply(taobao_filter_duplicates).reset_index(drop=True)
        buy = buy.drop(columns=['timeStamp_pv'])
        # Merge two datasets into aggregated raw data.
        buy = buy.rename(columns={'timeStamp': 'buyTime'})
        pv = pv.rename(columns={'timeStamp': 'pvTime'})
        buy.drop('Behavior', axis=1, inplace=True)
        pv.drop('Behavior', axis=1, inplace=True)
        raw = pv.merge(buy[['UserID', 'ProductID', 'Product_type_ID', 
                            'buyTime']], on=['UserID', 'ProductID', 
                                             'Product_type_ID'], how='left')
        # Manipulate timestamps and assign labels.
        earliest_time = raw[timestamp_col].min()
        raw[timestamp_col] = (raw[timestamp_col] - earliest_time).dt.total_seconds()
        raw['label'] = raw['buyTime'].notna().astype(int)
        raw['buyTime'] = (raw['buyTime'] - earliest_time).dt.total_seconds()
        # Get the time difference between conversions and clicks.
        raw['delay'] = np.where(raw['label'] == 1, 
                                raw['buyTime'] - raw[timestamp_col], -1)
        raw['delay_in_days'] = raw['delay'] / NUM_SEC_PER_DAY
        # Remove some rows based on certain conditions on selected columns.
        UserID_2_rmv = raw.UserID.value_counts().sort_values().head(150000).index.tolist()
        raw = raw[~raw['UserID'].isin(UserID_2_rmv)]
        ProductID_2_rmv = raw.ProductID.value_counts().sort_values().head(2000000).index.tolist()
        raw = raw[~raw['ProductID'].isin(ProductID_2_rmv)]
        Product_type_ID_2_rmv = raw.Product_type_ID.value_counts().sort_values().head(400).index.tolist()
        raw = raw[~raw['Product_type_ID'].isin(Product_type_ID_2_rmv)]
        ProductID_2_rmv = raw.ProductID.value_counts().sort_values().head(100).index.tolist()
        raw = raw[~raw['ProductID'].isin(ProductID_2_rmv)]
        # Get the unique values for all categorical columns.
        cat_dict = defaultdict(list)
        for col in cat_feats:
            unique_val = list(raw[col].unique())
            cat_dict[col] = unique_val
        # Apply encoding for all categorical columns.
        all_cat = defaultdict(create_empty_array)
        for col in cat_feats:
            all_cat[col] = get_ohe(cat_dict[col])
        for col in cat_feats:
            val_dict = all_cat[col]
            raw[col] = raw[col].apply(lambda x: val_dict[x])
        # Save the mid-point preprocessed dataset.
        raw.to_csv(f"{data_dir}/preprocessed_taobao.csv", index=False)
    # Get the statistics of the dataset.
    data_analytics(raw, timestamp_col)
    # Get the number of unique values for all categorical features into a list.
    total_cat = raw[cat_feats]
    cat_dims = [total_cat[col].unique().size for col in cat_feats]
    # Split the dataset into train and test.
    train_x, train_y, test_x, test_y = train_test_split(raw, train_start, 
                                                        train_end, num_test, 
                                                        timestamp_col)
    return train_x, train_y, test_x, test_y, cat_dims

