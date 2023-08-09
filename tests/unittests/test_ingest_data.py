from HousingPricePredictions import ingest_data
import os


def test_args():
    args = ingest_data.argument_parser()
    arguments = args.parse_args()
    # Test Case 1 Check if the path to store the raw data exists or not
    assert os.path.exists(arguments.raw_data_path)

    # Test Case 2 Check if the path to store the processed data exists or not
    assert os.path.exists(arguments.splitted_data_path)

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


def test_fetching_data():
    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = "data/raw"
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    ingest_data.fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH)
    assert os.path.exists(HOUSING_PATH + "/housing.csv")
    assert os.path.exists(os.path.join(HOUSING_PATH, "housing.tgz"))


def test_load_housing_data():
    HOUSING_PATH = "data/raw"

    try:
        # Load the file
        ingest_data.load_housing_data(HOUSING_PATH)
        assert os.path.exists(HOUSING_PATH + "/housing.csv")

    except Exception as e:
        # An exception should be raised
        assert type(e) == FileNotFoundError


def test_transform_and_split_data():

    splitted_data_path = "data/processed"

    assert os.path.exists(splitted_data_path + "/housing_prepared_train.csv")
    assert os.path.exists(splitted_data_path + "/strat_train_set.csv")
    assert os.path.exists(splitted_data_path + "/strat_test_set.csv")
