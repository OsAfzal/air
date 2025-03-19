import pandas as pd 
from glob import glob


def pm(path: str, drop=True):
    """
    This function reads all CSV files with a `.csv.gz` extension located within the 
    specified directory and its subdirectories, concatenates them into a single DataFrame, 
    and performs data cleaning and preprocessing.

    Parameters:
    -----------
    path : str
        The root directory path where the `.csv.gz` files are stored. The function will
        search recursively for files in subdirectories.
    
    drop : bool, optional, default=True
        Whether to drop certain columns ('location_id', 'sensors_id', 'lat', 'lon', 'parameter', 'units') 
        from the DataFrame. If set to False, these columns will be retained.

    Returns:
    --------
    pd.DataFrame
        A cleaned and processed DataFrame with the following transformations:
        - Concatenation of all CSV files in the specified directory.
        - Removal of rows with 'value' outside the range (0, 900).
        - Forward filling of missing values.
        - Conversion of 'datetime' column to pandas `datetime` format.
        - Setting 'datetime' as the index.

    Notes:
    ------
    - The function assumes that all input CSV files have a 'datetime' column.
    - The 'value' column is expected to contain numerical data.
    - The 'datetime' column is converted to pandas `datetime` and used as the index.
    - The following columns are dropped by default: 'location_id', 'sensors_id', 'lat', 'lon', 'parameter', 'units'.
    - The function uses forward filling (`ffill`) to handle missing values.

    Example:
    --------
    >>> df = pm('/path/to/data/')
    >>> df.head()
    """

    
    files = glob(f'{path}**/**/*.csv.gz', recursive=True)

    df_list = []

    for file in files:
        df = pd.read_csv(file, compression='gzip')
        df_list.append(df)

    df =  pd.concat(df_list, ignore_index=True)
    
    drop_cols = ['location_id', 'sensors_id', 'lat', 'lon', 'parameter', 'units', 'location']
    
    if drop:
        df.drop(columns=drop_cols, inplace=True)
    df = df[df['value'] > 0]
    df = df[df['value'] < 900]
    df.ffill(inplace=True)
    df = df.rename(columns={'value': 'pm2.5'})
        
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    return df