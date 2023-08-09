from HousingPricePredictions import train
import os


def test_args():

    args = train.argument_parser()
    arguments = args.parse_args()
    prpeared_data_path = arguments.prepared_data_path
    assert os.path.exists(prpeared_data_path)

    # Test Case 2 Check if the pickle path is prsent or not
    assert os.path.exists(arguments.pickle_path)

    # Test Case 3 Check if the path to store the log file is valid or not
    assert os.path.exists(arguments.log_file_path)

    # Test Case 4 Check if the console argument passed is either true or false
    assert arguments.console.lower() in ["true", "false"]

    # Test Case 5 Check if the log level argument passed is valid or not
    assert arguments.log_level.upper() in [
        "NOTSET",
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ]


def test_model_training():

    # Check if the housing_prepared.csv is present in desired location
    # processed_data_path = arguments.prepared_data_path
    pickle_path = "artifacts"
    # assert os.path.exists("data/processed/housing_prepared_train.csv")
    # train.model_training(arguments)

    # Check if the linear regression model is saved as pickle file in desired path
    assert os.path.exists(pickle_path + "/linear_reg_model.pkl")

    # Check if the Decision Tree regression  model is saved as pickle file in desired path
    assert os.path.exists(pickle_path + "/decision_tree_reg_model.pkl")


#     # Check if the Final model Regression is saved as pickle file in desired path
#     assert os.path.exists(pickle_path + "/final_reg_model.pkl")
