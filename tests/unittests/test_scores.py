import os

from HousingPricePredictions import scores


def test_args():

    args = scores.argument_parser()
    arguments = args.parse_args()

    # Test Case 1 Check if the model path exists or not
    assert os.path.exists(arguments.model_path)

    # Test Case 2 Check if the prepared data path is prsent or not
    assert os.path.exists(arguments.processed_data_path)

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


def test_validate_model():
    # Check if strat_test_set.csv is present in desired path
    model_path = "artifacts"
    processed_data_path = "data/processed"

    assert os.path.exists(processed_data_path + "/strat_test_set.csv")

    # Check if the final_model.pkl file is present in model path
    assert os.path.exists(model_path + "/final_reg_model.pkl")
