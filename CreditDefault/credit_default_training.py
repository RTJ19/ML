"""
Author: Roshan

The script takes in the data csv, preprocess,
trains and save the model in a pickle file.
"""

print ("Importing libraries....")

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import time
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import RandomOverSampler


class credit_default_training:

    def preprocess(self, df):
        """
        Cleans the raw data
        :param df: dataframe
        :return: Independent Variable x and
        Independent Variable Y
        """
        print("Started Processing....")
        # binary conversion
        df.replace(to_replace="yes", value=1, inplace=True)
        df.replace(to_replace="no", value=0, inplace=True)

        # replace unknowns with nan
        df = df.replace(to_replace="unknown", value=np.nan)
        # getting the list of columns with nan
        ml = df.columns[df.isna().any()].tolist()

        for item in ml:
            # getting the ratio of the index labels
            val = pd.DataFrame(df[item].value_counts(normalize=True))

            # index labels in a list
            valr = val.index.tolist()
            # drc.index = valr
            # columns values in a list
            valc = val[item].tolist()
            # replacing the nan values with ratio
            df[item] = df[item].fillna(pd.Series(np.random.choice(valr, p=valc, size=len(df))))

        # dependent variable
        dfy = df.iloc[:, -1]
        # independent variable
        dfx = df.iloc[:, :-1]

        # converting categorical data to numerical
        dfx = pd.get_dummies(dfx)

        # normalizing
        dfx = (dfx - dfx.min()) / (dfx.max() - dfx.min())

        dxdy = pd.concat([dfx, dfy], axis=1)

        # class balancing
        sm = RandomOverSampler(random_state=42)
        dfx, dfy = sm.fit_sample(dxdy.iloc[:, :-1], dxdy.iloc[:, -1])

        # converting to dataframe
        dfx = pd.DataFrame(dfx, columns=dxdy.iloc[:, :-1].columns.values)

        # dimensionality reduction
        pca = PCA(n_components=33)
        dfx = pca.fit_transform((dfx))

        print("Processing Done")

        return dfx, dfy

    def model_training(self, dfx, dfy):
        """
        Trains the model and saves it into a pickle file
        :param dfx: dataframe
        :param dfy: dataframe
        :return: pickle file
        """
        print("Started Training....")
        classifier = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                            max_depth=None, max_features='auto', max_leaf_nodes=None,
                                            min_impurity_decrease=0.0, min_impurity_split=None,
                                            min_samples_leaf=1, min_samples_split=2,
                                            min_weight_fraction_leaf=0.0, n_estimators=1000, n_jobs=1,
                                            oob_score=False, random_state=7, verbose=0, warm_start=False)

        classifier.fit(dfx, dfy)
        print("Training Done")

        print("Saving File to disk...")
        with open("credit_default_trained_model.pickle", 'wb') as fp:
            pickle.dump(classifier, fp)

        print("Done")


def main(data_file):
    print ("Reading File...")
    df = pd.read_csv(data_file)

    obj = credit_default_training()
    dfx, dfy = obj.preprocess(df)
    obj.model_training(dfx, dfy)


if __name__ == "__main__":
    start_time = time.time()
    main("credit.csv")
    print("---Script took %s seconds ---" % (time.time() - start_time))
