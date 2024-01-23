# Importing the libraries
import argparse
import os
import pickle
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from HousingPricePredictions.logger import configure_logger
import pandas as pd


def argument_parser():
    """
    This is the function to parse the arguments and return the arguments

    Parameters
    __________
    None

    Returns
    __________
    args

    Returns the parsed arguments that are passed in parser
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model_path",
        type=str,
        nargs="?",
        help="Path of the folder containing model pickle file",
        default="artifacts",
    )

    parser.add_argument(
        "processed_data_path",
        nargs="?",
        type=str,
        help="Path to the folder containing processed datasets",
        default="data/processed",
    )

    parser.add_argument(
        "-log",
        "--log_level",
        type=str,
        nargs="?",
        help="specify the logging level, default will be info",
        default="info",
    )

    parser.add_argument(
        "console",
        type=str,
        nargs="?",
        help="toggle whether or not to write logs to the console. Example: True or False",
        default="false",
    )

    parser.add_argument(
        "log_file_path",
        type=str,
        nargs="?",
        help="Provide the path for the log file",
        default="logs/scores_log.txt",
    )

    return parser


def validate_model(arguments):
    """
    The function to predict the output and score it

    Parameters
    __________
        Parameters are the command line arguments parsed arguments that contain:

    model_path : type str
                    Path to store the model pickle file

    processed_data_path : type str
                    Path to where the processed datas has to be stored

    log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"DEBUG"`

    console : str
                    Whether to console the log statments or not (True or False)

    log_file_path
                    Path to store the log file





    Returns
    ________
    None
    """
    processed_data_path = arguments.processed_data_path
    model_path = arguments.model_path
    log_level = arguments.log_level.upper()
    console = arguments.console.capitalize()
    log_file_path = arguments.log_file_path

    logging = configure_logger(
        log_file=log_file_path, console=console, log_level=log_level
    )

    validate_data = pd.read_csv(os.path.join(processed_data_path, "strat_test_set.csv"))
    logging.info(
        f"Reading and storing the validation data from the path {os.path.abspath(processed_data_path)}"
    )

    X_test = validate_data.drop("median_house_value", axis=1)
    y_test = validate_data["median_house_value"].copy()

    imputer = SimpleImputer(strategy="median")

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    imputer.fit(X_test_num)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )

    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    logging.info("Transformed the validation data")

    with open(os.path.join(model_path, "final_reg_model.pkl"), "rb") as file:
        final_reg_model = pickle.load(file)

    logging.info("Fetched and unpickeled the final model")

    logging.info("Predicting the values and printing scores")
    final_predictions = final_reg_model.predict(X_test_prepared)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    final_mae = mean_absolute_error(y_test, final_predictions)
    print("Root mean squared error of final reg model : ", final_rmse)
    print("Mean absolute error of final reg model : ", final_mae)

    logging.info("Done")


if __name__ == "__main__":
    arguments = argument_parser().parse_args()
    validate_model(arguments)
