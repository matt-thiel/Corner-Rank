import os
import re
import pandas as pd
from datetime import datetime
from config import (
    oos_alpha_dir,  oos_rsqrm_dir, oos_model_exposures_dir,
    oos_investable_universe_dir, oos_returns_dir,
    is_alpha_dir, is_rsqrm_dir, is_investable_universe_dir, is_returns_dir,
    full_alpha_dir, full_rsqrm_dir, full_investable_universe_dir, full_returns_dir
)
import csv
from io import StringIO
from utilities import clean_dataframe
import numpy as np


in_sample = False
out_of_sample = True

class DataRetrieval:
    """
    A class for retrieving various types of financial data from different directories.
    
    Attributes
    ----------
    alpha_dir : str
        Directory path for alpha data files.
    rsqrm_dir : str
        Directory path for risk model data files.
    investable_universe_dir : str
        Directory path for investable universe data files.
    returns_dir : str
        Directory path for returns data files.
    
    Methods
    -------
    get_alpha_dates()
        Retrieves dates of available alpha data files.
    get_rsqrm_betas_dates()
        Retrieves dates of available risk model beta data files.
    get_rsqrm_factor_dates()
        Retrieves dates of available risk model factor data files.
    get_rsqrm_correlations_dates()
        Retrieves dates of available risk model correlation data files.
    get_investable_universe_dates()
        Retrieves dates of available investable universe data files.
    get_returns_dates()
        Retrieves dates of available returns data files.
    load_alphas(date)
        Loads alpha data for a specific date.
    load_risk_betas(date)
        Loads risk model beta data for a specific date.
    load_investable_universe(date)
        Loads investable universe data for a specific date.
    load_returns_csv(date)
        Loads returns data for a specific date.
    load_risk_correlations(date)
        Loads risk model correlation data for a specific date.
    load_risk_factors(date)
        Loads risk model factor data for a specific date.
    """

    def __init__(self):
        if in_sample == True and out_of_sample == False:
            self.alpha_dir = is_alpha_dir
            self.rsqrm_dir = is_rsqrm_dir
            self.investable_universe_dir = is_investable_universe_dir
            self.returns_dir = is_returns_dir 

        elif in_sample == False and out_of_sample == True:
            self.alpha_dir = oos_alpha_dir
            self.rsqrm_dir = oos_rsqrm_dir
            self.investable_universe_dir = oos_investable_universe_dir
            self.returns_dir = oos_returns_dir 
            self.rsqrm_oos_dir = oos_model_exposures_dir

        elif in_sample == True and out_of_sample == True:
            self.alpha_dir = full_alpha_dir
            self.rsqrm_dir = full_rsqrm_dir
            self.investable_universe_dir = full_investable_universe_dir
            self.returns_dir = full_returns_dir 
            #self.rsqrm_oos_dir = rsqrm_oos_dir

    def get_alpha_dates(self):
        """
        Retrieves dates of available alpha data files.

        Returns
        -------
        list of datetime
            Sorted list of dates for which alpha data is available.
        """
        dates = set()
        for filename in os.listdir(self.alpha_dir):
            if filename.startswith('Alphas ') and filename.endswith('.txt'):
                date_pattern = re.compile(r'Alphas (\d{8})\.txt')
                match = date_pattern.search(filename)
                if match:
                    date_str = match.group(1)
                    try:
                        date = datetime.strptime(date_str, '%Y%m%d')
                        dates.add(date)
                    except ValueError:
                        print(f"Skipping file with invalid date format: {filename}")
        return sorted(dates)

    def get_rsqrm_betas_dates(self):
        """
        Retrieves dates of available risk model beta data files.

        Returns
        -------
        list of datetime
            Sorted list of dates for which risk model beta data is available.
        """
        dates = set()
        date_pattern = re.compile(r'RSQRM_US_v2_19_9g(\d{8})_USDcusip')
        for filename in os.listdir(self.rsqrm_dir):
            if filename.endswith('.csv'):
                match = date_pattern.search(filename)
                if match:
                    date_str = match.group(1)
                    try:
                        date = datetime.strptime(date_str, '%Y%m%d')
                        dates.add(date)
                    except ValueError:
                        print(f"Skipping file with invalid date format: {filename}")
        return sorted(dates)
    
    def get_rsqrm_betas_oos_dates(self):
        """
        Retrieves dates of available risk model beta data files.

        Returns
        -------
        list of datetime
            Sorted list of dates for which risk model beta data is available.
        """
        dates = set()
        date_pattern = re.compile(r'RSQRM_US_V2_19_9(\d{8})_USDc')
        for filename in os.listdir(self.rsqrm_oos_dir):
            if filename.endswith('.csv'):
                match = date_pattern.search(filename)
                if match:
                    date_str = match.group(1)
                    try:
                        date = datetime.strptime(date_str, '%Y%m%d')
                        dates.add(date)
                    except ValueError:
                        print(f"Skipping file with invalid date format: {filename}")
        return sorted(dates)
    
    def get_rsqrm_factor_dates(self):
        """
        Retrieves dates of available risk model factor data files.

        Returns
        -------
        list of datetime
            Sorted list of dates for which risk model factor data is available.
        """
        dates = set()
        date_pattern = re.compile(r'RSQRM_US_v2_19_9g_USD_(\d{8})_FactorDef')
        for filename in os.listdir(self.rsqrm_dir):
            if filename.endswith('.txt'):
                match = date_pattern.search(filename)
                if match:
                    date_str = match.group(1)
                    try:
                        date = datetime.strptime(date_str, '%Y%m%d')
                        dates.add(date)
                    except ValueError:
                        print(f"Skipping file with invalid date format: {filename}")
        return sorted(dates)

    def get_rsqrm_correlations_dates(self):
        """
        Retrieves dates of available risk model correlation data files.

        Returns
        -------
        list of datetime
            Sorted list of dates for which risk model correlation data is available.
        """
        dates = set()
        date_pattern = re.compile(r'RSQRM_US_v2_19_9g_USD_(\d{8})_Correl')
        for filename in os.listdir(self.rsqrm_dir):
            if filename.endswith('.txt'):
                match = date_pattern.search(filename)
                if match:
                    date_str = match.group(1)
                    try:
                        date = datetime.strptime(date_str, '%Y%m%d')
                        dates.add(date)
                    except ValueError:
                        print(f"Skipping file with invalid date format: {filename}")
        return sorted(dates)

    def get_investable_universe_dates(self):
        """
        Retrieves dates of available investable universe data files.

        Returns
        -------
        list of datetime
            Sorted list of dates for which investable universe data is available.
        """
        dates = set()
        date_pattern = re.compile(r'InvestableUniverse (\d{8})')
        for filename in os.listdir(self.investable_universe_dir):
            if filename.endswith('.txt'):
                match = date_pattern.search(filename)
                if match:
                    date_str = match.group(1)
                    try:
                        date = datetime.strptime(date_str, '%Y%m%d')
                        dates.add(date)
                    except ValueError:
                        print(f"Skipping file with invalid date format: {filename}")
        return sorted(dates)

    def get_returns_dates(self):
        """
        Retrieves dates of available returns data files.

        Returns
        -------
        list of datetime
            Sorted list of dates for which returns data is available.
        """
        dates = set()
        date_pattern = re.compile(r'Returns(\d{8})c\.csv')
        for filename in os.listdir(self.returns_dir):
            if filename.endswith('.csv'):
                match = date_pattern.search(filename)
                if match:
                    date_str = match.group(1)
                    try:
                        date = datetime.strptime(date_str, '%Y%m%d')
                        dates.add(date)
                    except ValueError:
                        print(f"Skipping file with invalid date format: {filename}")
        return sorted(dates)

    def load_alphas(self, date):
        """
        Loads alpha data for a specific date.

        Parameters
        ----------
        date : datetime
            The date for which to load alpha data.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame containing alpha data.

        Raises
        ------
        FileNotFoundError
            If no file is found for the specified date.
        """
        date_str = date.strftime('%Y%m%d')
        filename = f'Alphas {date_str}.txt'
        file_path = os.path.join(self.alpha_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for date: {date}")
        df = pd.read_csv(file_path, sep='|', header=0)
        return clean_dataframe(df)
        
    def load_risk_betas(self, date):
        """
        Loads risk model beta data for a specific date.

        Parameters
        ----------
        date : datetime
            The date for which to load risk model beta data.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame containing risk model beta data.

        Raises
        ------
        FileNotFoundError
            If no file is found for the specified date.
        """
        date_str = date.strftime('%Y%m%d')
        filename = f'RSQRM_US_v2_19_9g{date_str}_USDcusip.csv'
        file_path = os.path.join(self.rsqrm_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for date: {date}")
        
        factor_exposures = []
        with open(file_path, 'r', encoding='latin1') as file:
            csv_reader = csv.reader(file)
            for line in file:
                data = next(csv.reader(StringIO(line)))
                # Parse columns.
                cusip = data[0]
                value = data[3]
                monthly_residual_sd_pct = float(data[4])
                monthly_residual_decimal = float(data[4]) / 100.0
                betas = [float(data[i]) for i in range(6, 47)]
                currency_quotation = data[48]
                total_forecast_risk_ann_pct = data[49]
                
                row = {
                    'cusip': cusip,
                    'value': value,
                    'monthly_residual_sd_decimal': monthly_residual_decimal,
                    'RESIDUAL_PCT': monthly_residual_sd_pct,
                    'betas': betas,
                    'currency_quotation': currency_quotation,
                    'total_forecast_risk_ann_pct': total_forecast_risk_ann_pct
                }
                factor_exposures.append(row)
        
        return clean_dataframe(pd.DataFrame(factor_exposures))
    
    def load_risk_betas_OOS(self, date):
        """
        Loads risk model beta data for a specific date.

        Parameters
        ----------
        date : datetime
            The date for which to load risk model beta data.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame containing risk model beta data.

        Raises
        ------
        FileNotFoundError
            If no file is found for the specified date.
        """
        date_str = date.strftime('%Y%m%d')
        filename = f'RSQRM_US_V2_19_9{date_str}_USDc.csv'
        file_path = os.path.join(self.rsqrm_oos_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for date: {date}")
        
        factor_exposures = []
        with open(file_path, 'r', encoding='latin1') as file:
            csv_reader = csv.reader(file)
            for line in file:
                data = next(csv.reader(StringIO(line)))
                # Parse columns.
                cusip = data[0][:8]
                value = data[3]
                try:
                    monthly_residual_sd_pct = float(data[4])
                    monthly_residual_decimal = float(data[4]) / 100.0
            
                #
                
                
                    betas = [float(data[i]) for i in range(6, 46)] + [0.0]
                except ValueError:
                    print("skipping line")
                    print("file: ", filename)
                    #print("index: ", line)
                    continue
            
                
                    
                #currency_quotation = data[48]
                #total_forecast_risk_ann_pct = data[49]
                
                row = {
                    'cusip': cusip,
                    'value': value,
                    'monthly_residual_sd_decimal': monthly_residual_decimal,
                    'RESIDUAL_PCT': monthly_residual_sd_pct,
                    'betas': betas,
                    #'currency_quotation': currency_quotation,
                    #'total_forecast_risk_ann_pct': total_forecast_risk_ann_pct
                }
                factor_exposures.append(row)
        
        return clean_dataframe(pd.DataFrame(factor_exposures))
    
    def load_risk_betas_OOS_fix_missing_Final(self, date):
        """
        Loads risk model beta data for a specific date, attempting to fix missing or shifted data.
        Uses CUSIPs from all available universe files for comparison.

        Parameters
        ----------
        date : datetime
            The date for which to load risk model beta data.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame containing risk model beta data.

        Raises
        ------
        FileNotFoundError
            If no file is found for the specified date.
        """
        date_str = date.strftime('%Y%m%d')
        filename = f'RSQRM_US_V2_19_9{date_str}_USDc.csv'
        file_path = os.path.join(self.rsqrm_oos_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for date: {date}")
        
        # Get all investable universe dates
        investable_dates = self.get_investable_universe_dates()
        
        # Create a set of all CUSIPs from all universe files
        all_cusips = set()
        for universe_date in investable_dates:
            investable_universe = self.load_investable_universe(universe_date)
            all_cusips.update(investable_universe['cusip'])
        
        factor_exposures = []
        with open(file_path, 'r', encoding='latin1') as file:
            for line_num, line in enumerate(file, 1):
                data = next(csv.reader(StringIO(line)))
                for index, cell in enumerate(data):
                    cell_cusip = cell.strip()[:8]
                    if cell_cusip in all_cusips:
                        try:
                            company_name = data[index+1]
                            value = data[index+3]
                            if data[index+4] == '' or data[index+4] == None:
                                monthly_residual_sd_pct = np.nan
                                monthly_residual_decimal = np.nan
                            else:
                                monthly_residual_sd_pct = float(data[index+4])
                                monthly_residual_decimal = monthly_residual_sd_pct / 100.0
                            try:
                                betas = [float(data[i]) for i in range(index+6, index + 47)]
                            except ValueError:
                                betas = [float(data[i]) for i in range(index+6, index + 46)] + [0.0]
                            
                            row = {
                                'cusip': cell_cusip,
                                'company_name': company_name,
                                'value': value,
                                'monthly_residual_sd_decimal': monthly_residual_decimal,
                                'RESIDUAL_PCT': monthly_residual_sd_pct,
                                'betas': betas,
                            }
                            factor_exposures.append(row)
                            break
                        except (ValueError, IndexError):
                            print(f"Skipping line in file: {filename}")
                            print(f"Problematic line: {line}")
                            break
                
        print("number of missing cusips: ", len(all_cusips) - len(factor_exposures))
        return clean_dataframe(pd.DataFrame(factor_exposures))
    
    
    def load_investable_universe(self, date):
        """
        Loads investable universe data for a specific date.

        Parameters
        ----------
        date : datetime
            The date for which to load investable universe data.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame containing investable universe data.

        Raises
        ------
        FileNotFoundError
            If no file is found for the specified date.
        """
        date_str = date.strftime('%Y%m%d')
        filename = f'InvestableUniverse {date_str}.txt'
        file_path = os.path.join(self.investable_universe_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for date: {date}")
        df = pd.read_csv(file_path, header=0, delimiter='|')
        return clean_dataframe(df)

    def load_returns_csv(self, date):
        """
        Loads returns data for a specific date.

        Parameters
        ----------
        date : datetime
            The date for which to load returns data.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame containing returns data.

        Raises
        ------
        FileNotFoundError
            If no file is found for the specified date.
        """
        date_str = date.strftime('%Y%m%d')
        filename = f'Returns{date_str}c.csv'
        file_path = os.path.join(self.returns_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for date: {date}")
        df = pd.read_csv(file_path, header=0,  delimiter=',')
        df['Return'] = df['Return'] / 100
        df['Date'] = date
        df = df.rename(columns={'CUSIP': 'cusip'})
        df = df.set_index(['cusip', 'Date'])
        return clean_dataframe(df.sort_index())
    
    def load_risk_correlations(self, date):
        """
        Loads risk model correlation data for a specific date.

        Parameters
        ----------
        date : datetime
            The date for which to load risk model correlation data.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame containing risk model correlation data.

        Raises
        ------
        FileNotFoundError
            If no file is found for the specified date.
        """
        date_str = date.strftime('%Y%m%d')
        filename = f'FF_RSQ_RSQRM_US_v2_19_9g_USD_{date_str}_Correl.txt'
        file_path = os.path.join(self.rsqrm_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for date: {date}")
        df = pd.read_csv(file_path, header=2, delimiter='|', encoding='latin1')
        return clean_dataframe(df)
    
    def load_risk_factors(self, date):
        """
        Loads risk model factor data for a specific date.

        Parameters
        ----------
        date : datetime
            The date for which to load risk model factor data.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame containing risk model factor data.

        Raises
        ------
        FileNotFoundError
            If no file is found for the specified date.
        """
        date_str = date.strftime('%Y%m%d')
        filename = f'FF_RSQ_RSQRM_US_v2_19_9g_USD_{date_str}_FactorDef.txt'
        file_path = os.path.join(self.rsqrm_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found for date: {date}")
        df = pd.read_csv(file_path, header=2, delimiter='|', encoding='latin1')
        return clean_dataframe(df)