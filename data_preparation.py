from typing import List
import pandas as pd


# Function to prepare logs before import to main class
def convert_logs_to_start_complete_by_status(df: pd.DataFrame, status: str, timestamp: str,
                                             status_cols: List[str]) -> pd.DataFrame:
    """
    from column timestamp and status [complete, start] generate a new two column start_timestamp and end_timestamp
    """
    df['start_timestamp'] = df.loc[df[status] == status_cols[0], timestamp]
    df['complete_timestamp'] = df.loc[df[status] == status_cols[1], timestamp]
    df.drop([status, timestamp], axis=1, inplace=True)
    return df


def grouped_start_complete_timestamp_to_one_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    grouped two rows first like [NaT, complete_timestamp] and second like [start_timestamp, NaT]
    to one row [start_timestamp, complete_timestamp]
    """
    # choose column to be grouped - start_timestamp and complete_timestamp
    columns_grouped = [col for col in df.columns.values if (col != 'start_timestamp' and col != 'complete_timestamp')]
    grouped = df.groupby(columns_grouped, as_index=False)

    # Columns aggregation using first and last argument
    return grouped.agg({'start_timestamp': 'first', 'complete_timestamp': 'last'})


def prepare_logs(df: pd.DataFrame, status: str, timestamp: str, status_cols: List[str]) -> pd.DataFrame:
    return grouped_start_complete_timestamp_to_one_row(
        df=convert_logs_to_start_complete_by_status(df=df, status=status, timestamp=timestamp, status_cols=status_cols))
