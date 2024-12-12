import pandas as pd
import numpy as np
import os
from glob import glob
from datetime import datetime
import ast
from data_retrieval import DataRetrieval
from utilities import expand_betas, factor_names_ordered


def format_backtest_weights():
    """
    Format and save backtest weights from the most recent results file.

    This function loads the most recent backtest results, processes the weights,
    adds company names, and saves the formatted weights to a CSV file.

    Returns
    -------
    None

    Notes
    -----
    The function performs the following steps:
    1. Loads the most recent backtest results file.
    2. Processes each row of the results, extracting weights and CUSIPs.
    3. Creates a DataFrame with weights for each date.
    4. Adds company names to the weights DataFrame.
    5. Sorts the data by company name.
    6. Saves the formatted weights to a CSV file.
    """
    # Load the most recent result from the results folder
    results_folder = 'results'
    formatted_folder = 'formatted/weights'
    list_of_files = glob(os.path.join(results_folder, 'backtest_results_*.csv'))
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        backtest_results = pd.read_csv(latest_file)
        print(f"Loaded most recent results from: {latest_file}")
    else:
        print("No results files found in the results folder.")
        return

    # Convert date to datetime and sort
    backtest_results['date'] = pd.to_datetime(backtest_results['date'])
    backtest_results = backtest_results.sort_values('date')

    # Create a list to store weight data for each date
    weight_data_list = []
    all_cusips = set()

    # Process each row in the backtest results
    for _, row in backtest_results.iterrows():
        weights_string = row['weights']
        cusips = ast.literal_eval(row['cusips'])
        all_cusips.update(cusips)

        # Remove the newline characters and extra spaces
        weights_string = weights_string.replace('\n', '').replace('\'', '')

        # Convert the string to a numpy array and then to a list
        weights_array = np.fromstring(weights_string[1:-1], sep='  ')
        weights = [float(w) for w in weights_array]

        date = row['date'].strftime('%Y-%m-%d')

        try:
            df = pd.DataFrame({
                'CUSIP': cusips,
                str(date): weights,
            })
            df = df.set_index('CUSIP')
            weight_data_list.append(df)
        except Exception as e:
            print(f"Error creating DataFrame for date {date}: {str(e)}")
            print(f"num CUSIPs: {len(cusips)}")
            print(f"num Weights: {len(weights)}")

    # Concatenate all weight dataframes
    all_weights = pd.concat(weight_data_list, axis=1)

    # Load all investable universe files and create a CUSIP to name mapping
    data_retriever = DataRetrieval()
    investable_universe_dates = data_retriever.get_investable_universe_dates()
    
    cusip_name_dict = {}
    for date in investable_universe_dates:
        investable_universe = data_retriever.load_investable_universe(date)
        cusip_name_dict.update(dict(zip(investable_universe['cusip'], investable_universe['Name'])))

    # Add the company names to the all_weights dataframe
    all_weights['COMPANY NAME'] = all_weights.index.map(cusip_name_dict)

    # Reorder columns to have 'COMPANY NAME' first
    cols = ['COMPANY NAME'] + [col for col in all_weights.columns if col != 'COMPANY NAME']
    all_data = all_weights[cols]

    # Sort the dataframe by company name
    all_data = all_data.sort_values('COMPANY NAME')

    output_file =  f'formatted_weights_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    output_path = os.path.join(results_folder, formatted_folder, output_file)
    all_data.to_csv(output_path)

    print(f"Formatted weights saved to: {output_file}")

def format_portfolio_performance():
    """
    Format and save portfolio performance data from the most recent results file.

    This function loads the most recent portfolio performance data, resamples it to
    a four-weekly frequency, calculates various performance metrics, and saves the
    formatted data to a CSV file.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the formatted four-weekly portfolio performance data.

    Notes
    -----
    The function performs the following steps:
    1. Loads the most recent portfolio performance data file.
    2. Resamples the data to a four-weekly frequency.
    3. Calculates cumulative returns and other performance metrics.
    4. Transposes the data for easier viewing.
    5. Saves the formatted data to a CSV file.
    """
    # Load the CSV file
    csv_files = glob('results/*.csv')
    latest_file = max(csv_files, key=os.path.getctime)
    results_folder = 'results'
    formatted_folder = 'formatted/portfolio_performance'
    print(f"Loaded most recent results from: {latest_file}")
    performance_data = pd.read_csv(latest_file, index_col='date', parse_dates=True)

    # Generate the date range
    #date_range = pd.date_range(start='2017-12-27', end='2024-07-31', freq='4W-WED')
    date_range = pd.date_range(start='2014-08-13', end='2021-06-09', freq='4W-WED')

    # Initialize an empty list to store resampled data
    resampled_data_list = []

    # Manually resample the data
    for i in range(len(date_range) - 1):
        start_date = date_range[i]
        end_date = date_range[i+1]
        
        # Select data between current and next date in the range
        period_data = performance_data.loc[start_date:end_date]
        
        if not period_data.empty:
            # Aggregate the data for this period
            period_agg = {
                'gross_return_before_costs': (1 + period_data['gross_return_before_costs']).prod() - 1,
                'total_transaction_cost': (1 + period_data['total_transaction_cost']).prod() - 1,
                'total_short_cost': (1 + period_data['total_short_cost']).prod() - 1,
                'total_costs': (1 + period_data['total_costs']).prod() - 1,
                'total_value': period_data['total_value'].iloc[-1]
            }
            period_agg['date'] = end_date
            resampled_data_list.append(period_agg)

    # Convert the list of aggregated data to a DataFrame
    resampled_data = pd.DataFrame(resampled_data_list).set_index('date')

    # Calculate net return
    resampled_data['Net Return'] = resampled_data['gross_return_before_costs'] - resampled_data['total_costs']

    # Calculate cumulative returns
    resampled_data['Cumulative Net Return'] = (1 + resampled_data['Net Return']).cumprod() - 1
    resampled_data['Cumulative Gross Return'] = (1 + resampled_data['gross_return_before_costs']).cumprod() - 1

    # Rename columns for clarity
    resampled_data = resampled_data.rename(columns={
        'gross_return_before_costs': 'Gross Return',
        'total_transaction_cost': 'Transaction Costs',
        'total_short_cost': 'Short Costs',
        'total_costs': 'Total Costs',
        'total_value': 'Portfolio Value'
    })

    # Transpose the DataFrame to have dates as columns
    resampled_data_transposed = resampled_data.T

    # Format dates in column names
    resampled_data_transposed.columns = resampled_data_transposed.columns.strftime('%Y-%m-%d')

    # Reset index to make metrics a column
    resampled_data_transposed.reset_index(inplace=True)
    resampled_data_transposed = resampled_data_transposed.rename(columns={'index': 'Metric'})

    # Save to CSV
    output_file = f'four_weekly_portfolio_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    output_path = os.path.join(results_folder, formatted_folder, output_file)
    resampled_data_transposed.to_csv(output_path, index=False)
    

    print(f"Four-weekly portfolio performance data saved to: {output_file}")


    return resampled_data_transposed


def get_most_recent_data(date, data_dates, load_function):
    """
    Get the most recent data before or on a given date.

    Parameters
    ----------
    date : datetime
        The target date.
    data_dates : list of datetime
        Available data dates.
    load_function : callable
        Function to load data for a given date.

    Returns
    -------
    tuple
        A tuple containing the loaded data and the most recent date.
    """
    most_recent_date = max([d for d in data_dates if d <= date])
    return load_function(most_recent_date), most_recent_date


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

def format_betas_for_excel():
    """
    Format and save portfolio betas for Excel output.

    This function processes the most recent backtest results and formatted weights,
    calculates portfolio betas, and saves them in an Excel file.

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame containing the formatted portfolio betas, or None if processing fails.

    Notes
    -----
    The function performs the following steps:
    1. Loads the most recent backtest results and formatted weights.
    2. Calculates portfolio betas for each date.
    3. Filters and formats the betas for specific factors.
    4. Aligns the betas with predefined pbeta dates.
    5. Saves the formatted betas to an Excel file.
    """
    # Find the most recent backtest results file
    results_folder = 'results'
    formatted_folder = 'formatted/betas'
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

    
    # Define the factors we want to keep
    factors_to_keep = [
        'Euro', 'British Pound', 'Swiss Franc', 'Australian Dollar', 'Canadian Dollar', 'Japanese Yen',
        'US Market Large', 'US Market Small',
        'Energy Equipment & Services', 'Materials', 'Aerospace & Defence', 'Building & Construction',
        'Industrials', 'Transport', 'Consumer Discretionary', 'Retailers', 'Consumer Staples',
        'Health Care', 'Biotechnology & Pharmaceuticals', 'Banking', 'Diversified Financials',
        'Capital Markets', 'Insurance', 'Real Estate', 'Software & IT Services', 'Hardware & Technology',
        'Telecom Services', 'Utilities'
    ]
    

    #factors_to_keep = 

    # Extract and format the betas
    beta_data = []
    beta_meta = []
    portfolio_betas = []
    data_retriever = DataRetrieval()
    rsqrm_dates = data_retriever.get_rsqrm_betas_dates()

    for _, row in results_df.iterrows():
        evalDate = datetime.strptime(row['date'], '%Y-%m-%d')
        betaDate = datetime.strptime(row['risk_betas_date'], '%Y-%m-%d')
        cusips = eval(row['cusips'])  # Convert string representation of list to actual list
        stock_betas = eval(row['stock_betas'])  # Convert string representation of list to actual list
        if type(row['end_date']) != str:
            continue
        returnDate = datetime.strptime(row['end_date'], '%Y-%m-%d')
        risk_betas_df, risk_betas_date = get_most_recent_data(returnDate, rsqrm_dates, data_retriever.load_risk_betas)
        
        risk_betas_df = risk_betas_df[risk_betas_df['cusip'].isin(cusips)]
        # Get weights for this date
        date_str = evalDate.strftime('%Y-%m-%d')
        if date_str in weights_df.columns:
            weights = weights_df[date_str].dropna().to_dict()
        else:
            print(f"Warning: No weights found for date {date_str}")
            continue
        
        # Calculate portfolio betas
        portfolio_beta = np.zeros(len(stock_betas[0]))
        #for cusip, beta in zip(cusips, stock_betas):
        #    if cusip in weights:
        #        portfolio_beta += np.array(beta) * weights[cusip]
        testZip = zip(risk_betas_df['cusip'], risk_betas_df['betas'])
        #for idx, row in risk_betas_df.iterrows():
        #    if row['cusip'] in weights:
        #        portfolio_beta += np.array(row['betas']) * weights[row['cusip']]
        for cusip, beta in testZip:
            if cusip in weights:
                portfolio_beta += np.array(beta) * weights[cusip]
        
        portfolio_betas.append({
            'Date': evalDate,
            'betaDate': betaDate,
            'portfolio_betas': portfolio_beta.tolist(),
            'cusips': cusips  # Add cusips to portfolio_betas
        })
        
        for cusip, beta in zip(cusips, stock_betas):
            beta_data.append({
                'Date': evalDate,
                'betaDate': betaDate,
                'cusip': cusip,
                'betas': beta
            })
            beta_meta.append({
                'Date': evalDate,
                'betaDate': betaDate,
                'cusip': cusip,
            })

    # Create DataFrames from the beta data
    beta_df = pd.DataFrame(beta_data)
    beta_meta_df = pd.DataFrame(beta_meta)
    portfolio_betas_df = pd.DataFrame(portfolio_betas)

    beta_meta_df.set_index(['Date', 'cusip'], inplace=True)
    beta_df.set_index(['Date', 'cusip'], inplace=True)
    portfolio_betas_df.set_index('Date', inplace=True)

    # Expand portfolio betas
    portfolio_betas_expanded = expand_betas(portfolio_betas_df.rename(columns={'portfolio_betas': 'betas'}))
    portfolio_betas_expanded.columns = factor_names_ordered
    
    # Filter and transpose the DataFrame
    portfolio_betas_filtered = portfolio_betas_expanded[factors_to_keep].T
    portfolio_betas_filtered.index.name = 'Factor'
    
    # Generate pbeta dates
    pbeta_dates_df = generate_pbeta_dates()
    
    # Sort the portfolio betas by date
    portfolio_betas_filtered = portfolio_betas_filtered.sort_index(axis=1)
    
    # Function to get the most recent date
    def get_most_recent_date(target_date, available_dates):
        return max([date for date in available_dates if date <= target_date] or [None])
    
    # Create a new DataFrame with pbeta dates
    new_portfolio_betas = pd.DataFrame(index=portfolio_betas_filtered.index, columns=pbeta_dates_df.index)
    
    # Fill the new DataFrame with the most recent betas for each pbeta date
    for pbeta_date in pbeta_dates_df.index:
        most_recent_date = get_most_recent_date(pbeta_date, portfolio_betas_filtered.columns)
        if most_recent_date is not None:
            new_portfolio_betas[pbeta_date] = portfolio_betas_filtered[most_recent_date]
        else:
            print(f"Warning: No data available for or before {pbeta_date}")
    
    # Replace portfolio_betas_filtered with the new DataFrame
    portfolio_betas_filtered = new_portfolio_betas
    
    # Create an Excel file with the formatted betas
    output_filename = f"formatted_portfolio_betas_subset{latest_file.split('_')[2]}_{latest_file.split('_')[4].split('.')[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    output_path = os.path.join(results_folder, formatted_folder, output_filename)
    
    with pd.ExcelWriter(output_path) as writer:
        portfolio_betas_filtered.to_excel(writer, sheet_name='Portfolio Betas')

    print(f"Formatted betas saved to: {output_path}")
    return portfolio_betas_filtered

