import pandas as pd
from pandas.core.frame import DataFrame as DataFrame
from typing import Any
import json
import os
from dataclasses import dataclass, field


@dataclass
class ScriptConfig:
    config_file: str = "config.json"

    # Default configuration keys and their corresponding default values
    config_defaults: dict[str, Any] = field(
        default_factory=lambda: {
            "default_path": os.getcwd(),  # Default working path (location of data sets)
            "dataset_file_names": [],  # List containing strings representing file names of DFs
        }
    )

    # Config dictionary to hold loaded configuration
    config: dict[str, Any] = field(init=False)

    # This will hold the values for default_path, log_level, and timeout
    default_path: str = field(init=False)
    log_level: str = field(init=False)
    timeout: int = field(init=False)
    dataset_file_names: list[str] = field(init=False)

    def __post_init__(self):
        # Load the configuration when the object is created
        self.config = self.load_config()

        # Load the key values from config or default if they don't exist
        self.default_path = self.get_config_value("default_path")
        self.dataset_file_names = self.get_config_value("dataset_file_names")

    def load_config(self) -> dict[str, Any]:
        """
        Load the configuration file if it exists, otherwise return default values.

        Returns:
        - dict: Configuration or default values if the file does not exist or fails to load.
        """
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as file:
                    return json.load(file)
            except json.JSONDecodeError:
                print(
                    f"Error parsing {self.config_file}, using default configuration.")
                return self.config_defaults
        else:
            print(f"{self.config_file} not found, using default configuration.")
            return self.config_defaults

    def get_config_value(self, key: str) -> Any:
        """
        Retrieve a value from the config dictionary, defaulting to the value in config_defaults 
        if the key is not found.

        Args:
        - key (str): The key to retrieve from the config.

        Returns:
        - Any: The value associated with the key, or the default value if the key is not found.
        """
        return self.config.get(key, self.config_defaults[key])


# Example usage:

# Create an instance of the ScriptConfig class
script_config = ScriptConfig()

# Access the config values directly from the class instance
print(f"Using default path: {script_config.default_path}")


def load_datasets(base_path: str =script_config.default_path,
                  script_config_: ScriptConfig = script_config) -> DataFrame:
    dfs: list[DataFrame] = []
    for df in script_config_.dataset_file_names:
        dfs.append(pd.read_csv(f"{base_path}/{df}", delimiter=";"))
    return dfs


# Set outliers to None for all numerical columns in the DataFrame
def remove_outliers_IQR(col: pd.Series, limits: tuple[float, float] = (0.25, 0.75)) -> pd.Series:
    """ Remove outliers for all numerical columns in the df"""
    # Ensure the input column is numeric
    if col.dtype not in ['float64', 'int64']:
        raise TypeError("Input column must be numeric.")

    Q1: float = col.quantile(limits[0])
    Q3: float = col.quantile(limits[1])
    IQR: float = Q3 - Q1

    # Filter out the outliers and set them to None
    return col.where((col >= (Q1 - 1.5 * IQR)) & (col <= (Q3 + 1.5 * IQR)))


def main():
    # Load the three data sets
    dfs: list[DataFrame] = load_datasets()

    # Merge the datasets using 'id_audit' as the common key
    df_merged = pd.merge(
        dfs[0], dfs[1], on="id_audit", how="inner"
    )  # First merge temp and power
    df_merged = pd.merge(
        df_merged, dfs[2], on="id_audit", how="inner"
    )  # Then merge with radio

    print(df_merged)


if __name__ == "__main__":
    main()
