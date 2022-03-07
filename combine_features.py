import pandas as pd

from pathlib import Path

from global_vars import FEATURES_FILEPATH


def impact_features_df():
    f_lst = {}
    df_lst = []

    files_csv = Path(FEATURES_FILEPATH).rglob('*.csv')
    # files_xlsx = Path(FILEPATH).rglob('*.xlsx')

    df_merged = pd.DataFrame()
    counter = 0

    for file in list(files_csv)[:]:
        df = pd.read_csv(file, index_col=0, header=0)
        df.index.name = 'tech'
        df = (df.reset_index().drop_duplicates(subset='tech', keep='last').set_index('tech').sort_index())
        df_lst.append(df)
        f_lst[file.name[:-4]] = df.copy()
        df = df.add_suffix('__' + file.name)

        if counter != 0:
            df_merged = pd.merge(df_merged, df, right_index=True, left_index=True, how='outer')
        else:
            df_merged = df
        counter += 1

    df_copy = df_merged.copy()
    df_copy.fillna(0, inplace=True)
    df_copy = df_copy.loc[:, (df_copy != 0).any(axis=0)]
    return df_copy
