"""
Admin code resolution utilities for handling column name aliases.
"""
from typing import List
import pandas as pd
import numpy as np


def resolve_admin_code(df: pd.DataFrame, aliases: List[str] = None) -> pd.Series:
    """
    Resolve admin code column from a DataFrame using a list of possible aliases.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame potentially containing admin code column
    aliases : List[str], optional
        List of possible column names for admin codes, in order of preference.
        If None, uses default ['FEWSNET_admin_code', 'admin_code', 'adm_code']

    Returns
    -------
    pd.Series
        Series containing admin codes

    Raises
    ------
    KeyError
        If no admin code column is found in the DataFrame

    Examples
    --------
    >>> df = pd.DataFrame({'FEWSNET_admin_code': [1, 2, 3], 'value': [10, 20, 30]})
    >>> admin_codes = resolve_admin_code(df)
    >>> print(admin_codes.tolist())
    [1, 2, 3]
    """
    if aliases is None:
        aliases = ['FEWSNET_admin_code', 'admin_code', 'adm_code']

    for col in aliases:
        if col in df.columns:
            s = df[col]
            if s.notna().any():
                return s

    raise KeyError(
        f"No admin code column found. Searched for: {aliases}. "
        f"Available columns: {df.columns.tolist()}"
    )


def resolve_admin_code_array(df: pd.DataFrame, aliases: List[str] = None) -> np.ndarray:
    """
    Resolve admin code column and return as numpy array.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame potentially containing admin code column
    aliases : List[str], optional
        List of possible column names for admin codes, in order of preference.
        If None, uses default ['FEWSNET_admin_code', 'admin_code', 'adm_code']

    Returns
    -------
    np.ndarray
        Array containing admin codes

    Raises
    ------
    KeyError
        If no admin code column is found in the DataFrame
    """
    series = resolve_admin_code(df, aliases)
    return series.values
