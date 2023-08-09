# importing the required libraries

import argparse
import tarfile
import os
import pandas as pd
from HousingPricePredictions.logger import configure_logger
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.impute import SimpleImputer
from six.moves import urllib


def argument_parser():
    """
    This is the function to parse the arguments and return the arguments

    Parameters
    _____________
    None

    Returns
    ___________
    args

    Returns the parsed arguments that are passed in parser
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "raw_data_path",
        type=str,
        help="Specifying the path to store the fetched raw housing dataset",
        nargs="?",
        default="data/raw",
    )

    parser.add_argument(
        "splitted_data_path",
        type=str,
        help="Specifying the path to store the processed housing dataset",
        nargs="?",
        default="data/processed",
    )

    parser.add_argument(
        "-log",
        "--log_level",
        type=str,
        help="Specify the logging level, default will be info",
        nargs="?",
        default="info",
    )

    parser.add_argument(
        "log_file_path",
        type=str,
        help="Provide the path for log file",
        nargs="?",
        default="logs/ingest_data_log.txt",
    )

    parser.add_argument(
        "console",
        type=str,
        help="toggle whether or not to write logs to the console. Example: True or False",
        nargs="?",
        default="false",
    )

    return parser


arguments = argument_parser()


def fetch_housing_data(housing_url, housing_path):
    """
    This function is to fetch the housing data

    Parameters
    __________
    housing_url : type str
    Path from which the housing data to be fetched

    housing_path : type str
    Path to which the housing data to be stored


    Returns
    ___________
    None
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path):
    """
    Function to load the housing datasets

    Parameters
    ___________
    housing_path : type str
    Path from which the data to be fetched and loaded

    Returns
    _________
    None
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def income_cat_proportions(data):
    """
    Function to return the proportions of income categories

    Parameters
    __________
    data : Dataframe
    Dataframe

    Returns
    ___________
    type float
    I
    Income category proportions
    """
    return data["income_cat"].value_counts() / len(data)


def transform_and_split_data(arguments):
    """
    Function to transform the data and split the data and stores the data

    Parameters
    ______________
    Parameters are the parsed command line arguments that contains:

    raw_data_path : type str
                    Path to store the fetched raw data from URL

    splitted_data_path : type str
                    Path to where the processed datas has to be stored

    log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"DEBUG"`

    log_file_path
                    Path to store the log file

    console : str
                    Whether to console the log statments or not (True or False)


    Returns
    _____________
    None
    """

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    log_file = arguments.log_file_path
    console_toggle = arguments.console.capitalize()
    log_level = arguments.log_level.upper()
    logging = configure_logger(
        log_file=log_file, log_level=log_level, console=console_toggle
    )

    splitted_data_path = arguments.splitted_data_path
    HOUSING_PATH = arguments.raw_data_path
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    logging.info("Housing data from the URL is fetched and stored in the housing_path")

    housing = load_housing_data(HOUSING_PATH)
    logging.info(f"Loaded the housing data from {os.path.abspath(HOUSING_PATH)}")

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    logging.info("Stratified Splitting the data into train and test set ")

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    logging.info("Random splitting the data into train and test set")

    train_set.to_csv(os.path.join(splitted_data_path, "train_set.csv"), index=False)
    test_set.to_csv(os.path.join(splitted_data_path, "test_set.csv"), index=False)

    logging.info(
        f"Downloaded the Randomly splitted tarin set and test set at {os.path.abspath(splitted_data_path)}"
    )

    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()
    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    strat_train_set.to_csv(
        os.path.join(splitted_data_path, "strat_train_set.csv"), index=False
    )
    strat_test_set.to_csv(
        os.path.join(splitted_data_path, "strat_test_set.csv"), index=False
    )

    logging.info(
        f"Downloaded the Stratified train data and Stratified test data {os.path.abspath(splitted_data_path)}"
    )

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.copy()  # drop labels for training set

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    housing_prepared.to_csv(
        os.path.join(splitted_data_path, "housing_prepared_train.csv"), index=False
    )
    logging.info(
        f"Downloaded the final prepared train dataset at {os.path.abspath(splitted_data_path)}"
    )


if __name__ == "__main__":
    arguments = argument_parser().parse_args()
    transform_and_split_data(arguments)
