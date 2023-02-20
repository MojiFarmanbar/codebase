import pandas as pd
import numpy as np
from bank.data import import_config


def categorical_feature_mapper(dataframe: pd.DataFrame) -> pd.DataFrame:
    """loads data and process categorical
    """

    # loading data

    df_raw = dataframe.copy(deep=True)
    cat_col = df_raw.select_dtypes(include="object")

    # loading config
    config = import_config()
    config_categorical_features = config["categorical_features"]
    config_cyclical_features = config["cyclical_features"]

    #
    cat_col = cat_col.assign(
        job=lambda df: df["job"].replace(config_categorical_features["job"]),
        marital=lambda df: df["marital"].replace(
            config_categorical_features["marital"]
        ),
        education=lambda df: df["education"].replace(
            config_categorical_features["education"]
        ),
        housing=lambda df: df["housing"].replace(
            config_categorical_features["housing"]
        ),
        loan=lambda df: df["loan"].replace(config_categorical_features["loan"]),
        y=lambda df: df["y"].replace(config_categorical_features["y"]),
        contact=lambda df: df["contact"].replace(
            config_categorical_features["contact"]
        ),
        month1=lambda df: df["month"].replace(config_cyclical_features["months"]),
        day_of_week1=lambda df: df["day_of_week"].replace(
            config_cyclical_features["days"]
        ),
    )

    cat_col["day_of_week1"] = pd.to_numeric(cat_col["day_of_week1"])
    cat_col["day_of_week1" + "_sin"] = np.sin(2 * np.pi * cat_col["day_of_week1"] / 7)
    cat_col["month1" + "_cos"] = np.cos(2 * np.pi * cat_col["month1"] / 12)
    return cat_col


def numerical_feature_mapper(dataframe: pd.DataFrame) -> pd.DataFrame:
    """loading numerical features and process them
    """
    # loading data
    df_raw = dataframe.copy(deep=True)
    num_col = df_raw.select_dtypes(include=["int", "float"])

    # loading config
    config = import_config()
    config_numerical_features = config["numerical_features"]

    num_col = num_col.assign(
        log_duration=lambda df: (
            df[config_numerical_features["log_duration"]] + 1
        ).transform(np.log),
        log_campaign=lambda df: (
            df[config_numerical_features["log_campaign"]] + 1
        ).transform(np.log),
    )

    return num_col
