# Importing the libraries
import argparse
import os
from scipy.stats import randint
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from HousingPricePredictions.logger import configure_logger
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression


def argument_parser():
    """
    This is the function to parse the arguments and return the arguments

    Parameters
    ___________
    None

    Returns
    ________
    args

    Returns the parsed arguments that are passed in parser
    """

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "prepared_data_path",
        type=str,
        help="Path to fetch the prepared data",
        nargs="?",
        default="data/processed",
    )

    parser.add_argument(
        "pickle_path",
        type=str,
        help="The path to save the pickle file",
        nargs="?",
        const=1,
        default="artifacts",
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
        nargs="?",
        help="Specify the path of the log file",
        default="logs/train_log.txt",
    )

    parser.add_argument(
        "console",
        type=str,
        nargs="?",
        help="toggle whether or not to write logs to the console. Example: True or False",
        default="false",
    )

    return parser


def model_training(arguments):
    """
    This is the function to train the module by using stored splitted data

    Parameters
    ___________
    Parameters are the command line arguments parsed arguments that contain:

    prepared_data_path : type str
                    Path to where the processed datas has to be stored

    pickle_path : type str
                    Path to store the model pickle file


    log_level: str
                    One of `["INFO","DEBUG","WARNING","ERROR","CRITICAL"]`
                    default - `"DEBUG"`

    log_file_path
                    Path to store the log file

    console : str
                    Whether to console the log statments or not (True or False)


    Returns
    ____________
    None

    """
    prepared_data_path = arguments.prepared_data_path
    log_file = arguments.log_file_path
    console = arguments.console.capitalize()
    log_level = arguments.log_level.upper()

    logging = configure_logger(log_file=log_file, console=console, log_level=log_level)

    path = os.path.join(prepared_data_path, "housing_prepared_train.csv")
    housing_prepared = pd.read_csv(path)
    logging.info(
        f"Loaded the prepared dataset from {os.path.abspath(prepared_data_path)}"
    )

    housing_labels = housing_prepared["median_house_value"].copy()
    housing_prepared = housing_prepared.drop("median_house_value", axis=1)

    lin_reg = LinearRegression()
    logging.info("Training the Linear Regression model")
    lin_reg.fit(housing_prepared, housing_labels)

    with open(
        os.path.join(arguments.pickle_path, "linear_reg_model.pkl"), "wb"
    ) as file1:
        pickle.dump(lin_reg, file1)

    logging.info("Saving the linear regression model")

    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse = mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("Linear Regression model root mean squared error : ", lin_rmse)
    lin_mae = mean_absolute_error(housing_labels, housing_predictions)
    print("Linear Regression model mean absolute error :", lin_mae)

    tree_reg = DecisionTreeRegressor(random_state=42)
    logging.info("Training the Decision Tree Regression model")
    tree_reg.fit(housing_prepared, housing_labels)

    with open(
        os.path.join(arguments.pickle_path, "decision_tree_reg_model.pkl"), "wb"
    ) as file2:
        pickle.dump(tree_reg, file2)

    logging.info("Saving the Decision Tree regression model")

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print("Decision Tree regression model root mean squared error :", tree_rmse)
    tree_mae = mean_absolute_error(housing_labels, housing_predictions)
    print("Decision Tree Regression model mean absolute error :", tree_mae)

    param_distribs = {
        "n_estimators": randint(low=1, high=200),
        "max_features": randint(low=1, high=8),
    }

    forest_reg = RandomForestRegressor(random_state=42)
    rnd_search = RandomizedSearchCV(
        forest_reg,
        param_distributions=param_distribs,
        n_iter=10,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )
    logging.info(
        "Building Random Forest Regression model with randomized search on hyper parameter"
    )

    rnd_search.fit(housing_prepared, housing_labels)
    cvres = rnd_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    param_grid = [
        # try 12 (3×4) combinations of hyperparameters
        {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
        # then try 6 (2×3) combinations with bootstrap set as False
        {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    # train across 5 folds, that's a total of (12+6)*5=90 rounds of training
    grid_search = GridSearchCV(
        forest_reg,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        return_train_score=True,
    )

    grid_search.fit(housing_prepared, housing_labels)
    logging.info(
        "Building Random Forest Regression model with grid search on hyper parameters"
    )

    grid_search.best_params_
    cvres = grid_search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)

    feature_importances = grid_search.best_estimator_.feature_importances_
    sorted(zip(feature_importances, housing_prepared.columns), reverse=True)

    final_model = grid_search.best_estimator_
    logging.info("Finding the best estimator model")

    with open(
        os.path.join(arguments.pickle_path, "final_reg_model.pkl"), "wb"
    ) as file3:
        pickle.dump(final_model, file3)

    logging.info("Saved the best estimated final model")


if __name__ == "__main__":
    arguments = argument_parser().parse_args()
    model_training(arguments)
