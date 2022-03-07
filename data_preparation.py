import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from global_vars import LABEL0, LABEL1


def arrange_emerg_data(df: pd.DataFrame):
    return df[df[['Time prediction range']].apply(lambda x: x[0].isdigit(), axis=1)]


def merge_features_and_emerg(df_impact: pd.DataFrame, df_emerg: pd.DataFrame):
    df = pd.merge(df_impact, df_emerg, left_index=True, right_index=True, how='right')
    df = df.iloc[:, :-df_emerg.shape[1]]
    df = df[~df.index.duplicated(keep='first')]

    df['label'] = None
    for anc in LABEL0:
        df.at[anc, 'label'] = 0
    for anc in LABEL1:
        df.at[anc, 'label'] = 1
    return df


def data_preparation(df: pd.DataFrame):
    X = df.iloc[:, :-1]
    cor_matrix = X.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    df1 = X.drop(to_drop, axis=1)

    df1['label'] = df['label']
    df1['label'] = df1['label'].fillna(-11111)
    df1.fillna(0, inplace=True)

    df_labeled = df1.loc[(df1['label'] == 1) | (df1['label'] == 0)]
    df_unlabeled = df1.loc[(df1['label'] == -11111)]
    return df_labeled, df_unlabeled


def split_data(df_labeled: pd.DataFrame):
    X = df_labeled.iloc[:, :-1]
    y = df_labeled.iloc[:, -1]

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)
