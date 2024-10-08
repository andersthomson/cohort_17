import pandas as pd
import numpy as np
from tabulate import tabulate
from pandas.core.frame import DataFrame as DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

DATASET_PATH = '/repo/ehrlcuj/cohort_17'


def peek_df(df: DataFrame, num: int = 5):
    """ Peek into the 'num' first rows in the data frame and check for empty entries """
    print(df.isnull().sum())
    print(tabulate(df.head(num), headers='keys', tablefmt='pretty'))


def filter_df(df: DataFrame, column_name: str, value: list[str | int]) -> DataFrame:
    """
    Return a dataframe with only rows that match the column/value
    Example:
        filter_df(df, 'id', [147, 200, 4]) -> DF with only rows containing id = 147 | 200 | 4
    """
    res = df[df[column_name].isin(value)]
    assert isinstance(res, DataFrame)
    return res


def bar_graph(
    df_: DataFrame, x: str, y: str, title: str = "", ylabel: str = "", xlabel: str = "",
    figsize: tuple[int, int] = (12, 6), width: float = 0.6, color: str = 'skyblue',
    convert_numeric: bool = True, head: int = 500
) -> None:
    """
    Plots a bar graph using the specified columns from a pandas DataFrame.

    Parameters
    ----------
   df : The pandas DataFrame containing the data to plot.
    x : The column name to use for the x-axis values.
    y : The column name to use for the y-axis values.
    title : The title of the bar graph (default is an empty string).
    ylabel : The label for the y-axis (default is an empty string).
    xlabel : The label for the x-axis (default is an empty string).
    figsize : The size of the figure in the format (width, height) (default is (12, 6)).
    width : The width of the bars in the bar graph (default is 0.6).
    color : The color of the bars in the bar graph (default is 'skyblue').
    convert_numeric : Whether to convert the x and y values to numeric types (default is True).
    head : The number of rows to include from the DataFrame (default is 500).

    Raises
    ------
    TypeError
        If the input `df` is not a pandas DataFrame.
    ValueError
        If the specified columns `x` or `y` do not exist in the DataFrame.

    Returns
    -------
    None
        This function does not return a value. It displays a bar plot.

    Example
    -------
    >>> data = {'id_audit': [147, 148, 149, 150], 'value': [23.5, 25.6, 22.1, 24.3]}
    >>> df = pd.DataFrame(data)
    >>> bar_graph(df, x='id_audit', y='value', title='Audit Temperatures', ylabel='Temperature [C]')
    """
    if x not in df_.columns or y not in df_.columns:
        s: str = f"Columns {x} and/or {y} not found in DataFrame"
        raise ValueError(s)

    # Extract only the relevant columns
    df = df_[[x, y]]

    # Optional: Convert to numeric if needed
    if convert_numeric:
        df[x] = pd.to_numeric(df[x], errors='coerce')
        df[y] = pd.to_numeric(df[y], errors='coerce')

    # Optionally take the first `head` rows
    df = df.head(head)

    # Create the plot
    plt.figure(figsize=figsize)
    plt.bar(df[x].astype(str), df[y], width=width, color=color)

    # Set labels and title
    plt.xlabel(xlabel if xlabel else x)
    plt.ylabel(ylabel if ylabel else y)
    plt.title(title)

    # Show the plot
    plt.show()


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


def main() -> None:
    # Load the three data sets
    dfs: dict[str, DataFrame] = {
        'temp': pd.read_csv(
            f'{DATASET_PATH}/dataset_1_radioTemperatures_20210303.csv', delimiter=';'
        ),
        'power': pd.read_csv(f'{DATASET_PATH}/dataset_2_powerClass_20210303.csv', delimiter=';'),
        'radio': pd.read_csv(f'{DATASET_PATH}/dataset_3_radioId_20210303.csv', delimiter=';')
    }

    # Peek into the n first rows of each data frame
    for df in dfs:
        peek_df(dfs[df], num=3)

    # Show summary of the 'temp' set
    pd.set_option('display.max_columns', 500)  #we want to be able to see all columns
    print(dfs['temp'].describe())

    # Plot
    # Extract values from id_audit == n
    id_147: DataFrame = filter_df(dfs['temp'], column_name='id_audit', value=[2199])
    x = id_147['value']
    plt.plot(x)
    bar_graph(
        dfs['temp'], x='id_audit', y='value', xlabel="ID Audit", ylabel="Temperature [C]", head=500
    )

    # Show outliers
    # sns.boxplot(x=dfs['temp']['value'])
    # plt.show()
    # Remove outliers using IQR method (Interquartile range)
    dfs['temp']['value'] = pd.to_numeric(dfs['temp']['value'], errors='coerce')
    dfs['temp']['value'] = remove_outliers_IQR(dfs['temp']['value'])
    print(dfs['temp'].describe())
    # sns.boxplot(x=dfs['temp']['value'])
    # plt.show()

    # Show temperatures per id_audit and sensor
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6))
    axes[0].scatter(x=dfs['temp']['id_audit'], y=dfs['temp']['value'], edgecolors='b')
    axes[1].scatter(x=dfs['temp']['field'], y=dfs['temp']['value'], edgecolors='b')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

