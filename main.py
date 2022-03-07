import pandas as pd
from combine_features import impact_features_df

from global_vars import *
from data_preparation import merge_features_and_emerg, arrange_emerg_data, data_preparation
from alg import logit_regression_iterative_process, logit_scores


def main():
    # path = 'https://drive.google.com/uc?export=download&id=' + EMERGING_FILE_URL.split('/')[-2]
    # df_emerg = pd.read_csv(path)
    df_emerg = arrange_emerg_data(pd.read_csv(EMERGING_FILE_PATH, index_col=0))
    df_features = impact_features_df()
    df = merge_features_and_emerg(df_features, df_emerg)
    df_labeled, df_unlabeled = data_preparation(df)
    indices, coeffs = logit_regression_iterative_process(df_labeled, df_unlabeled)
    logit_scores(df, indices, coeffs)


if __name__ == '__main__':
    main()
