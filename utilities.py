import numpy as np
import pandas as pd

def clean_dataframe(df):
    return df.replace([np.inf, -np.inf], np.nan).dropna()


def expand_betas(beta_df):    
# First, let's create a new DataFrame from the 'betas' column
    beta_df_expanded = pd.DataFrame(beta_df['betas'].tolist(), index=beta_df.index)
    num_factors = len(beta_df_expanded.columns)

    beta_df_expanded.columns = [f'beta_{i+1}' for i in range(num_factors)]

    return beta_df_expanded

factor_names_ordered = ["Euro", "British Pound", "Swiss Franc", "Australian Dollar", "Canadian Dollar", "Japanese Yen", "US Market Large", "US Market Small", "Energy Equipment & Services", "Materials", "Aerospace & Defence", "Building & Construction", "Industrials", "Transport", "Consumer Discretionary", "Retailers", "Consumer Staples", "Health Care", "Biotechnology & Pharmaceuticals", "Banking", "Diversified Financials", "Capital Markets", "Insurance", "Real Estate", "Software & IT Services", "Hardware & Technology", "Telecom Services", "Utilities", 
                        "Stat1", "Stat2", "Stat3", "Stat4", "Stat5", "Stat6", "Stat7", "Stat8", "Stat9", "Stat10", "Stat11", "Stat12", "Stat13"]
factor_names_ordered_OOS = [
    "Euro",
    "British Pound",
    "Swiss Franc",
    "Australian Dollar",
    "Canadian Dollar",
    "Japanese Yen",
    "Dividend Yield",
    "Value",
    "Growth Trend",
    "Growth Momentum",
    "Short Term Price Momentum",
    "Long Term Price Momentum",
    "Leverage",
    "Liquidity",
    "Quality",
    "US Market Large",
    "US Market Small",
    "Energy Equipment & Services",
    "Materials",
    "Aerospace & Defence",
    "Building & Construction",
    "Industrials",
    "Transport",
    "Consumer Discretionary",
    "Retailers",
    "Consumer Staples",
    "Health Care",
    "Biotechnology & Pharmaceuticals",
    "Banking",
    "Diversified Financials",
    "Capital Markets",
    "Insurance",
    "Real Estate",
    "Software & IT Services",
    "Hardware & Technology",
    "Telecom Services",
    "Utilities",
    "Statistical factor1",
    "Statistical factor2",
    "Statistical factor3",
    "Statistical factor4"
]