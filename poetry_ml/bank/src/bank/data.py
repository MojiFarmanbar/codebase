import logging
from pathlib import Path
import pandas as pd
import yaml
from yaml.loader import SafeLoader
from typing import Dict


def import_config(file_path: Path = './config/config.yml') -> Dict:
    """importing config file

    Parameters
    ----------
    path : str
        Path to data
    Returns
    -------
    df : Dict
    """
    path = Path(__file__).parent / file_path
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data


def load_data(path: Path) -> pd.DataFrame:
    """Load the data and convert the column names.

    Parameters
    ----------
    path : str
        Path to data
    Returns
    -------
    df : pandas.DataFrame
        DataFrame with data
    """
    logger = logging.getLogger(__name__)
    logger.info("Reading data from %s", path)
    df = pd.read_csv(path, delimiter=";").rename(columns=str.lower)
    logger.info("Read %i rows", len(df))
    return df
