import pandas as pd
import numpy as np
import os
import os.path as osp

from data_formatters import ETTFormatter


def load_ozone_data(data_dir):
    save_path = osp.join(data_dir, 'ozone_processed.csv')
    if osp.exists(save_path):
        return pd.read_csv(save_path)

    def process_single(df: pd.DataFrame):
        # for label in output:
        #     srs = output[label]

        df['year'] = df['year'].astype(str)
        df['month'] = df['month'].apply(lambda x: '{:02d}'.format(x))
        df['day'] = df['day'].apply(lambda x: '{:02d}'.format(x))
        df['hour'] = df['hour'].apply(lambda x: '{:02d}'.format(x))
        df['date'] = df['year'] + '-' + df['month'] + '-' + df['day'] + ' ' + df['hour'] + ':00'
        df['date'] = pd.to_datetime(df['date'])

        df.index = df['date']
        earliest_time = df.index.min()

        start_date = min(df.fillna(method='ffill').dropna().index)
        end_date = max(df.fillna(method='bfill').dropna().index)

        active_range = (df.index >= start_date) & (df.index <= end_date)
        df = df[active_range].fillna(method='bfill')
        date = df.index

        df['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (date - earliest_time).days * 24
        df['days_from_start'] = (date - earliest_time).days
        df['hour'] = date.hour
        df['day'] = date.day
        df['day_of_week'] = date.dayofweek
        df['month'] = date.month
        df['year'] = date.year

        df.drop(columns=['No'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    df_list = []

    for i, filename in enumerate(os.listdir(data_dir)):
        if filename.split('.')[-1] != 'csv':
            continue
        df_single = pd.read_csv(osp.join(data_dir, filename))
        # create id column
        df_single['id'] = i
        # process each Dataframe
        df_single = process_single(df_single.copy())
        df_list.append(df_single)

    # concat all
    df = pd.concat(df_list, axis=0)
    # save file
    df.to_csv(save_path)

    return df


def load_electricity_data(data_dir):
    save_path = osp.join(data_dir, 'hourly_elect_processed.csv')
    if osp.exists(save_path):
        return pd.read_csv(save_path)

    csv_path = osp.join(data_dir, 'LD2011_2014.txt')

    df = pd.read_csv(csv_path, index_col=0, sep=';', decimal=',')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    output = df.resample('1h').mean().replace(0, np.nan)

    earliest_time = output.index.min()

    df_list = []
    for label in output:
        srs = output[label]

        start_date = min(srs.fillna(method='ffill').dropna().index)
        end_date = max(srs.fillna(method='bfill').dropna().index)

        active_range = (srs.index >= start_date) & (srs.index <= end_date)
        srs = srs[active_range].fillna(0.0)

        tmp = pd.DataFrame({'power_usage': srs})
        date = tmp.index
        tmp['t'] = (date - earliest_time).seconds / 60 / 60 + (date - earliest_time).days * 24
        tmp['days_from_start'] = (date - earliest_time).days
        tmp['categorical_id'] = label
        tmp['date'] = date
        tmp['id'] = label
        tmp['hour'] = date.hour
        tmp['day'] = date.day
        tmp['day_of_week'] = date.dayofweek
        tmp['month'] = date.month

        df_list.append(tmp)

    output = pd.concat(df_list, axis=0, join='outer').reset_index(drop=True)

    output['categorical_id'] = output['id'].copy()
    output['hours_from_start'] = output['t']
    output['categorical_day_of_week'] = output['day_of_week'].copy()
    output['categorical_hour'] = output['hour'].copy()

    # Filter to match range used by other academic papers
    output = output[(output['days_from_start'] >= 1096) & (output['days_from_start'] < 1346)].copy()

    output.to_csv(save_path)

    return output


def load_ett_data(data_dir, filename):
    csv_path = os.path.join(data_dir, filename)

    df = pd.read_csv(csv_path)
    df.date = pd.to_datetime(df.date)
    df.index = df.date

    earliest_time = df.date.min()
    date = df.index

    df['id'] = 0
    df['hours_from_start'] = (date - earliest_time).seconds / 60 / 60 + (date - earliest_time).days * 24
    df['days_from_start'] = (date - earliest_time).days
    df['hour'] = date.hour
    df['day'] = date.day
    df['day_of_week'] = date.dayofweek
    df['month'] = date.month

    df.reset_index(drop=True, inplace=True)

    return df[df['days_from_start'] < 600]


if __name__ == '__main__':
    df = load_ett_data('data/ETT-small', 'ETTh1.csv')
    formatter = ETTFormatter()
    train, valid, test = formatter.split_data(df, L_in=96)
