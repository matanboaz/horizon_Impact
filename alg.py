import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from data_preparation import split_data


def logit_regression_iterative_process(df_labeled: pd.DataFrame, df_unlabeled: pd.DataFrame, threshold=0.99):
    X_train, X_test, y_train, y_test = split_data(df_labeled)
    X_unlabeled = df_unlabeled.iloc[:, :-1]

    # Initiate iteration counter
    iterations = 0

    # Containers to hold f1_scores and # of pseudo-labels
    train_f1s = []
    test_f1s = []
    pseudo_labels = []

    # Containers to hold the feature column numbers and its corresponding coefficient
    coeffs = []
    features_indices = []
    dct = {}

    # Assign value to initiate while loop
    high_prob = [1]

    # Loop will run until there are no more high-probability pseudo-labels
    while len(high_prob) > 0:
        # Fit classifier and make train/test predictions
        clf = LogisticRegression(max_iter=1000,
                                 penalty='l1',
                                 solver='liblinear',
                                 C=0.1)
        clf.fit(X_train, y_train)
        y_hat_train = clf.predict(X_train)
        y_hat_test = clf.predict(X_test)

        # Calculate iteration # and f1 scores, and store f1 scores
        train_f1 = f1_score(y_train, y_hat_train)
        test_f1 = f1_score(y_test, y_hat_test)
        train_f1s.append(train_f1)
        test_f1s.append(test_f1)
        print(f'Iteration {iterations}')
        print(f'Train f1: {train_f1}')
        print(f'Test f1: {test_f1}')

        # Generate predictions and probabilities for unlabeled data
        print(f'Now predicting labels for unlabeled data...')

        pred_probs = clf.predict_proba(X_unlabeled)
        preds = clf.predict(X_unlabeled)
        prob_0 = pred_probs[:, 0]
        prob_1 = pred_probs[:, 1]

        # Store predictions and probabilities in dataframe
        df_pred_prob = pd.DataFrame([])
        df_pred_prob['preds'] = preds
        df_pred_prob['prob_0'] = prob_0
        df_pred_prob['prob_1'] = prob_1
        df_pred_prob.index = X_unlabeled.index

        # Separate predictions with > threshold probability
        high_prob = pd.concat(
            [
                df_pred_prob.loc[df_pred_prob['prob_0'] > threshold],
                df_pred_prob.loc[df_pred_prob['prob_1'] > threshold]
            ],
            axis=0
        )
        print(f'{len(high_prob)} high=probability predictions added to training data.')

        pseudo_labels.append((len(high_prob)))

        # Add pseudo-labeled data to training data
        X_train = pd.concat([X_train, X_unlabeled.loc[high_prob.index]], axis=0)
        y_train = pd.concat([y_train, high_prob.preds])

        # Drop pseudo-labeled instances from unlableld data
        X_unlabeled = X_unlabeled.drop(index=high_prob.index)
        print(f'{len(X_unlabeled)} unlabeled instances remaining.\n')

        # Store iteration features and coefficients
        temp_dict = {}
        for i, coef in enumerate(clf.coef_.flatten()):
            if np.abs(coef) > 0.0:
                if i in features_indices:
                    continue
                else:
                    features_indices.append(i)
                    coeffs.append(coef)
        temp_dict = {str(iterations): features_indices}
        dct[str(iterations)] = temp_dict

        # Update iteration counter
        iterations += 1

    return features_indices, coeffs


def logit_scores(df: pd.DataFrame, indices: list, coeffs: list):
    df_chosen_cols = pd.DataFrame()
    for ind in indices:
        df_chosen_cols[df.columns[ind]] = df.iloc[:, ind]
    print(df_chosen_cols)
