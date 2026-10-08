import pandas as pd


def analyze_column_types(df):
    """Handle analyze column types."""
    column_types = {}

    for column in df.columns:
        dtype = df[column].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            unique_ratio = len(df[column].unique()) / len(df)
            if unique_ratio < 0.05:
                column_types[column] = "categorical"
            else:
                if pd.api.types.is_integer_dtype(dtype):
                    column_types[column] = "discrete"
                else:
                    column_types[column] = "continuous"

        elif pd.api.types.is_datetime64_dtype(dtype):
            column_types[column] = "datetime"

        elif pd.api.types.is_bool_dtype(dtype):
            column_types[column] = "boolean"

        else:
            unique_ratio = len(df[column].unique()) / len(df)
            if unique_ratio < 0.05:
                column_types[column] = "categorical"
            else:
                column_types[column] = "text"

    return column_types
