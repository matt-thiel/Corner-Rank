import pandas as pd
import numpy as np
import os
from datetime import datetime
from utilities import expand_betas, factor_names_ordered_OOS
from config import oos_model_exposures_dir
import re
from data_retrieval import DataRetrieval
from glob import glob

class RSQRMBetaFormatter:
    @staticmethod
    def get_most_recent_date(target_date, available_dates):
        return max([date for date in available_dates if date <= target_date] or [None])

    def __init__(self):
        self.results_folder = 'results'
        self.formatted_folder = 'formatted/model_exposure_performance'
        self.factors_to_keep = [
            "Euro", "British Pound", "Swiss Franc", "Australian Dollar", "Canadian Dollar", "Japanese Yen",
            "Dividend Yield", "Value", "Growth Trend", "Growth Momentum", "Short Term Price Momentum",
            "Long Term Price Momentum", "Leverage", "Liquidity", "Quality", "US Market Large", "US Market Small",
            "Energy Equipment & Services", "Materials", "Aerospace & Defence", "Building & Construction",
            "Industrials", "Transport", "Consumer Discretionary", "Retailers", "Consumer Staples", "Health Care",
            "Biotechnology & Pharmaceuticals", "Banking", "Diversified Financials", "Capital Markets", "Insurance",
            "Real Estate", "Software & IT Services", "Hardware & Technology", "Telecom Services", "Utilities",
            "Statistical factor1", "Statistical factor2", "Statistical factor3", "Statistical factor4"
        ]
        self.data_retrieval = DataRetrieval()

    def format_rsqrm_betas(self, date):
        file_name = f"RSQRM_US_V2_19_9{date.strftime('%Y%m%d')}_USDc.csv"
        file_path = os.path.join(oos_model_exposures_dir, file_name)

        #date_str = date.strftime('%Y%m%d')
        
        df = self.data_retrieval.load_risk_betas_OOS_fix_missing_Final(date)
        #betas = df.iloc[:, 4:45]
        #betas = df.iloc[:, 6:47]
        betas = df['betas']
        betas.index = df.iloc[:, 0]
        #betas_transposed = betas.T
        #factor_names = [f'Factor_{i+1}' for i in range(41)]
        #betas_transposed.index = factor_names
        
        return betas

    def process_rsqrm_betas(self):
        results_folder = 'results'
         #formatted_folder = 'formatted/betas'
        csv_files = [f for f in os.listdir(results_folder) if f.startswith('backtest_results_') and f.endswith('.csv')]
        if not csv_files:
            print("No backtest results found.")
            return None

        latest_file = max(csv_files, key=lambda f: os.path.getmtime(os.path.join(results_folder, f)))
        file_path = os.path.join(results_folder, latest_file)

        # Load the backtest results
        results_df = pd.read_csv(file_path)

        # Load the formatted weights
        weights_files = glob(os.path.join(results_folder, 'formatted/weights', 'formatted_weights_*.csv'))
        if not weights_files:
            print("No formatted weights file found.")
            return None
        latest_weights_file = max(weights_files, key=os.path.getmtime)
        weights_df = pd.read_csv(latest_weights_file, index_col='CUSIP')


        rsqrm_files = [f for f in os.listdir(oos_model_exposures_dir) if f.startswith('RSQRM_US_V2_19_9') and f.endswith('_USDc.csv')]
        if not rsqrm_files:
            print("No RSQRM files found.")
            return None

        all_betas = []
        date_pattern = re.compile(r'RSQRM_US_V2_19_9(\d{8})_USDc')
        for file in rsqrm_files:
            match = date_pattern.search(file)
            if match:
                date_str = match.group(1)
                #try:
                date = datetime.strptime(date_str, '%Y%m%d')
                betas = self.format_rsqrm_betas(date)
                all_betas.append({'Date': date, 'betas': betas})
                #except ValueError:
                    #print(f"Skipping file with invalid date format: {file}")
        all_betas_df = pd.DataFrame(all_betas)
        all_betas_df.sort_index(inplace=True)

        portfolio_betas = []

        for _, row in results_df.iterrows():
            #evalDate = datetime.strptime(row['date'], '%Y-%m-%d')
            #betaDate = datetime.strptime(row['risk_betas_date'], '%Y-%m-%d')
            if type(row['start_date']) != str:
                continue
            betaDate = datetime.strptime(row['start_date'], '%Y-%m-%d')
            returnDate = datetime.strptime(row['date'], '%Y-%m-%d')

            #print(f"Processing date: {evalDate}")
            cusips = eval(row['cusips'])
            weights_df_filtered = weights_df.fillna(0)
            weights_df_filtered = weights_df_filtered.loc[cusips]
            weights_df_filtered = weights_df_filtered.fillna(0)

            date_str = betaDate.strftime('%Y-%m-%d')
            if date_str in weights_df.columns:
                weights = weights_df_filtered[date_str].dropna().to_dict()
            else:
                print(f"Warning: No weights found for date {date_str}")
                continue

            # Find the most recent date to betaDate
            most_recent_date = all_betas_df['Date'][all_betas_df['Date'] <= betaDate].max()
            if pd.notnull(most_recent_date):
                betas = all_betas_df.loc[all_betas_df['Date'] == most_recent_date, 'betas'].iloc[0]
                betas = pd.DataFrame(betas).reset_index()
                betas['cusip'] = betas['cusip'].str[:8]
                betas = betas[betas['cusip'].isin(cusips)]
                # Calculate portfolio betas
                if len(betas) != len(cusips):
                    print(most_recent_date)
                    print(f"Warning: Mismatch in lengths - betas: {len(betas)}, cusips: {len(cusips)}")
                    print(f'mismatching cusips: {[x for x in cusips if x not in betas["cusip"].tolist()]}')
                portfolio_beta = np.zeros(len(betas['betas'].iloc[0]))
                for cusip, beta in zip(betas['cusip'], betas['betas']):
                    if cusip in weights:
                        portfolio_beta += np.array(beta) * weights[cusip]
                
                portfolio_betas.append({
                    'Date': returnDate,
                    'betaDate': betaDate,
                    'portfolio_betas': portfolio_beta.tolist(),
        })
                # Process betas here
            else:
                print(f"No beta data found for or before {betaDate}")


        betas_df = pd.DataFrame(all_betas)
        betas_df.set_index('Date', inplace=True)

        # Expand betas
        #betas_expanded = expand_betas(betas_df)
        #betas_expanded.columns = factor_names_ordered_OOS

        # Create DataFrame from the portfolio betas
        portfolio_betas_df = pd.DataFrame(portfolio_betas)
        portfolio_betas_df.set_index('Date', inplace=True)

        # Expand portfolio betas
        portfolio_betas_expanded = expand_betas(portfolio_betas_df.rename(columns={'portfolio_betas': 'betas'}))
        portfolio_betas_expanded.columns = factor_names_ordered_OOS
        
        # Filter and transpose the DataFrame
        portfolio_betas_filtered = portfolio_betas_expanded[self.factors_to_keep].T
        portfolio_betas_filtered.index.name = 'Factor'

        pbeta_dates_df = generate_pbeta_dates()
        # Sort the portfolio betas by date
        portfolio_betas_filtered = portfolio_betas_filtered.sort_index(axis=1)

        # Create a new DataFrame with pbeta dates
        new_portfolio_betas = pd.DataFrame(index=portfolio_betas_filtered.index, columns=pbeta_dates_df.index)
        
        # Fill the new DataFrame with the most recent betas for each pbeta date
        for pbeta_date in pbeta_dates_df.index:
            most_recent_date = self.get_most_recent_date(pbeta_date, portfolio_betas_filtered.columns)
            if most_recent_date is not None:
                new_portfolio_betas[pbeta_date] = portfolio_betas_filtered[most_recent_date]
            else:
                print(f"Warning: No data available for or before {pbeta_date}")
         # Fill NaN values in portfolio_betas_filtered with zero
        #new_portfolio_betas = new_portfolio_betas.fillna(0)
        new_portfolio_betas.fillna(0, inplace=True)
        # Replace portfolio_betas_filtered with the new DataFrame
        portfolio_betas_filtered = new_portfolio_betas

        # Create an Excel file with the formatted betas
        output_filename = f"formatted_rsqrm_betas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        output_path = os.path.join(self.results_folder, self.formatted_folder, output_filename)

        with pd.ExcelWriter(output_path) as writer:
            portfolio_betas_filtered.to_excel(writer, sheet_name='Portfolio Betas')

        print(f"Formatted daily betas saved to: {output_path}")
        return portfolio_betas_filtered


def generate_pbeta_dates():
    """
    Generate a DataFrame of specific pbeta dates.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with a DatetimeIndex of pbeta dates.
    """
    '''
    # List of specific pbeta dates
    pbeta_dates = [
        '27-Dec-17', '24-Jan-18', '21-Feb-18', '21-Mar-18', '18-Apr-18', '16-May-18',
        '13-Jun-18', '11-Jul-18', '8-Aug-18', '5-Sep-18', '3-Oct-18', '31-Oct-18',
        '28-Nov-18', '26-Dec-18', '23-Jan-19', '20-Feb-19', '20-Mar-19', '17-Apr-19',
        '15-May-19', '12-Jun-19', '10-Jul-19', '7-Aug-19', '4-Sep-19', '2-Oct-19',
        '30-Oct-19', '27-Nov-19', '25-Dec-19', '22-Jan-20', '19-Feb-20', '18-Mar-20',
        '15-Apr-20', '13-May-20', '10-Jun-20', '8-Jul-20', '5-Aug-20', '2-Sep-20',
        '30-Sep-20', '28-Oct-20', '25-Nov-20', '23-Dec-20', '20-Jan-21', '17-Feb-21',
        '17-Mar-21', '14-Apr-21', '12-May-21', '9-Jun-21', '7-Jul-21', '4-Aug-21',
        '1-Sep-21', '29-Sep-21', '27-Oct-21', '24-Nov-21', '22-Dec-21', '19-Jan-22',
        '16-Feb-22', '16-Mar-22', '13-Apr-22', '11-May-22', '8-Jun-22', '6-Jul-22',
        '3-Aug-22', '31-Aug-22', '28-Sep-22', '26-Oct-22', '23-Nov-22', '21-Dec-22',
        '18-Jan-23', '15-Feb-23', '15-Mar-23', '12-Apr-23', '10-May-23', '7-Jun-23',
        '5-Jul-23', '2-Aug-23', '30-Aug-23', '27-Sep-23', '25-Oct-23', '22-Nov-23',
        '20-Dec-23', '17-Jan-24', '14-Feb-24', '13-Mar-24', '10-Apr-24', '8-May-24',
        '5-Jun-24', '3-Jul-24', '31-Jul-24'
    ]
    '''
    pbeta_dates_OOS = [
    "13-Aug-14", "10-Sep-14", "8-Oct-14", "5-Nov-14", "3-Dec-14", "31-Dec-14",
    "28-Jan-15", "25-Feb-15", "25-Mar-15", "22-Apr-15", "20-May-15", "17-Jun-15",
    "15-Jul-15", "12-Aug-15", "9-Sep-15", "7-Oct-15", "4-Nov-15", "2-Dec-15",
    "30-Dec-15", "27-Jan-16", "24-Feb-16", "23-Mar-16", "20-Apr-16", "18-May-16",
    "15-Jun-16", "13-Jul-16", "10-Aug-16", "7-Sep-16", "5-Oct-16", "2-Nov-16",
    "30-Nov-16", "28-Dec-16", "25-Jan-17", "22-Feb-17", "22-Mar-17", "19-Apr-17",
    "17-May-17", "14-Jun-17", "12-Jul-17", "9-Aug-17", "6-Sep-17", "4-Oct-17",
    "1-Nov-17", "29-Nov-17", "27-Dec-17", "24-Jan-18", "21-Feb-18", "21-Mar-18",
    "18-Apr-18", "16-May-18", "13-Jun-18", "11-Jul-18", "8-Aug-18", "5-Sep-18",
    "3-Oct-18", "31-Oct-18", "28-Nov-18", "26-Dec-18", "23-Jan-19", "20-Feb-19",
    "20-Mar-19", "17-Apr-19", "15-May-19", "12-Jun-19", "10-Jul-19", "7-Aug-19",
    "4-Sep-19", "2-Oct-19", "30-Oct-19", "27-Nov-19", "25-Dec-19", "22-Jan-20",
    "19-Feb-20", "18-Mar-20", "15-Apr-20", "13-May-20", "10-Jun-20", "8-Jul-20",
    "5-Aug-20", "2-Sep-20", "30-Sep-20", "28-Oct-20", "25-Nov-20", "23-Dec-20",
    "20-Jan-21", "17-Feb-21", "17-Mar-21", "14-Apr-21", "12-May-21", "9-Jun-21"
    ]
    
    # Convert string dates to datetime objects
    pbeta_dates = [datetime.strptime(date, '%d-%b-%y') for date in pbeta_dates_OOS]
    
    # Create a DataFrame with the pbeta dates
    df = pd.DataFrame({'Date': pbeta_dates})
    df.set_index('Date', inplace=True)
    
    return df



def main():
    print("Starting RSQRM beta formatting process...")
    formatter = RSQRMBetaFormatter()
    formatted_betas = formatter.process_rsqrm_betas()
    if formatted_betas is not None:
        print("RSQRM beta formatting completed successfully.")
    else:
        print("RSQRM beta formatting failed.")

if __name__ == "__main__":
    main()