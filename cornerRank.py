import pandas as pd
import numpy as np
from data_retrieval import DataRetrieval
from asset_selection import AlphaReturnGridBuilder, AssetSelector
from utilities import expand_betas
from scipy.optimize import minimize
from datetime import datetime
import sys, os


class CornerRank:
    """
    A class for implementing corner rank portfolio optimization strategy.
    
    This class implements a portfolio optimization strategy that selects assets based on 
    their alpha characteristics and optimizes weights while maintaining factor neutrality.
    
    Parameters
    ----------
    start_date : str
        Start date for the backtest in 'YYYY-MM-DD' format
    end_date : str
        End date for the backtest in 'YYYY-MM-DD' format
    n_day_returns : int, optional
        Number of days to use for returns calculation, by default 7
    max_returns_days : int, optional
        Maximum number of days of returns to load at each rebalance, by default 25
    alpha_bins : int, optional
        Number of bins to use for alpha selection grid, by default 30
        
    Attributes
    ----------
    data_retriever : DataRetrieval
        Instance of DataRetrieval class for loading market data
    results : list
        List to store backtest results
    active_long : pd.Series
        Series containing active long positions
    active_short : pd.Series
        Series containing active short positions
    opt_count : int
        Counter for optimization iterations
    last_portfolio : pd.DataFrame
        DataFrame containing the most recent portfolio weights
    """
    def __init__(self, start_date, end_date, n_day_returns=7, max_returns_days=25, alpha_bins=30):
        # Convert input dates to datetime objects
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Configuration parameters
        self.n_day_returns = n_day_returns  # Number of days for returns calculation
        self.max_returns_days = max_returns_days  # Maximum days of returns to load
        self.alpha_bins = alpha_bins  # Number of bins for alpha selection
        
        # Initialize data retriever and results storage
        self.data_retriever = DataRetrieval()
        self.results = []
        
        # Initialize portfolio tracking variables
        self.active_long = pd.Series(dtype=str)
        self.active_short = pd.Series(dtype=str)
        self.opt_count = 0
        
        # Initialize last portfolio state variables
        self.last_portfolio = pd.DataFrame(columns=['cusip', 'weights'])
        self.last_opt_weights = None
        self.last_transaction_costs = None
        self.last_portfolio_names = None
        self.last_betas = None
        
        # Set random seed for reproducibility
        np.random.seed(16)

            def get_most_recent_data(self, date, data_dates, load_function):
        """
        Retrieves the most recent data prior to a given date.
        
        Parameters
        ----------
        date : datetime
            Target date for data retrieval
        data_dates : list
            List of available data dates
        load_function : callable
            Function to load data for a specific date
            
        Returns
        -------
        tuple
            (loaded_data, most_recent_date)
        """
        most_recent_date = max([d for d in data_dates if d <= date])
        return load_function(most_recent_date), most_recent_date
    
    def get_transaction_short_costs(self, invested_caps, max_cap, min_cap):
        """
        Calculates transaction and short costs based on market capitalization.
        
        Parameters
        ----------
        invested_caps : pd.DataFrame
            DataFrame containing market cap information for invested assets
        max_cap : float
            Maximum market cap in the universe
        min_cap : float
            Minimum market cap in the universe
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing transaction and short costs for each asset
        """
        # Calculate costs using logarithmic scaling based on market cap
        return invested_caps.apply(
            lambda row: pd.Series({
                'cusip': row['cusip'],
                # Transaction cost decreases with market cap
                'trade_cost': 20 - 15 * (np.log(row['Trade.Cost.Mkt.Cap']) - np.log(min_cap)) / (np.log(max_cap) - np.log(min_cap)),
                # Short cost decreases with market cap
                'short_cost': (100 - 80 * (np.log(row['Trade.Cost.Mkt.Cap']) - np.log(min_cap)) / (np.log(max_cap) - np.log(min_cap))) 
            }),
            axis=1
        )

    def update_returns_data(self, current_date, returns_dates):
        """
        Updates the returns data for the current rebalance date.
        
        Parameters
        ----------
        current_date : datetime
            Current rebalance date
        returns_dates : list
            List of available returns dates
            
        Returns
        -------
        pd.DataFrame
            Updated returns data for the relevant period
        """
        # Get the most recent dates up to max_returns_days
        relevant_dates = sorted([d for d in returns_dates if d <= current_date], reverse=True)[:self.max_returns_days]
        
        # Load and concatenate returns data
        returns_data = pd.concat([self.data_retriever.load_returns_csv(date) for date in relevant_dates])
        return returns_data.sort_values('Date').groupby('cusip').tail(self.max_returns_days)

    def get_weekly_returns(self, start_date, end_date, cusips):
        """
        Calculates weekly returns for specified assets between two dates.
        
        Parameters
        ----------
        start_date : datetime
            Start date for returns calculation
        end_date : datetime
            End date for returns calculation
        cusips : list
            List of asset identifiers
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing weekly returns for specified assets
        """
        # Get relevant dates within the specified range
        returns_dates = [d for d in self.data_retriever.get_returns_dates() if start_date < d <= end_date]
        
        # Initialize returns DataFrame
        daily_returns = pd.DataFrame(index=cusips)
        
        # Load daily returns for each date
        for date in returns_dates:
            return_frame = self.data_retriever.load_returns_csv(date)
            daily_returns[date] = return_frame.loc[return_frame.index.get_level_values('cusip').isin(cusips), 'Return'].droplevel('Date')
        
        # Fill missing values and calculate compound returns
        daily_returns = daily_returns.fillna(0)
        weekly_returns = (1 + daily_returns).prod(axis=1) - 1
        
        return pd.DataFrame(weekly_returns, columns=['Return'])

    def calculate_portfolio_returns(self, results_df):
        """
        Calculates portfolio returns including transaction costs and short costs.
        
        Parameters
        ----------
        results_df : pd.DataFrame
            DataFrame containing portfolio weights and transaction data
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing portfolio returns and costs
        """
        portfolio_returns = []
        dates = sorted(results_df['date'].tolist())
        total_value = 100000000  # Initialize portfolio value to 100 million

        for i, current_date in enumerate(dates):
            # Handle first date separately - only calculate costs
            if i == 0:
                transaction_costs = results_df[results_df['date'] == current_date]['transaction_costs'].iloc[0]
                weight_changes = results_df[results_df['date'] == current_date]['weights'].iloc[0]
                
                # Calculate annualization factor for costs
                days_between_rebalance = 5
                annualization_factor = days_between_rebalance / 252
                
                # Get transaction costs for initial portfolio
                transaction_costs = results_df[results_df['date'] == current_date]['transaction_costs'].iloc[0]
                previous_cusips = results_df[results_df['date'] == current_date]['cusips'].iloc[0]
                new_cusips = []

                # Combine weights and costs information
                all_weights = pd.DataFrame({
                    'cusip': previous_cusips,
                    'weight_change': weight_changes
                })
                all_weights = pd.merge(all_weights, transaction_costs, on='cusip', how='left')

                # Calculate short costs for short positions
                all_weights['short_cost'] = np.where(
                    all_weights['weight_change'] < 0,
                    (all_weights['short_cost'] / 10000) * np.abs(all_weights['weight_change']) * annualization_factor,
                    0)
                
                # Calculate transaction costs
                all_weights['transaction_cost'] = (all_weights['trade_cost'] / 10000) * np.abs(all_weights['weight_change'])

                # Sum up total costs
                total_transaction_cost = all_weights['transaction_cost'].sum()
                total_short_cost = all_weights['short_cost'].sum() 
                total_costs = total_transaction_cost + total_short_cost

                # Record initial portfolio state
                portfolio_returns.append({
                    'start_date': None,
                    'end_date': current_date,
                    'portfolio_return_after_costs': 0,
                    'total_transaction_cost': -1*total_transaction_cost,
                    'total_short_cost': -1*total_short_cost,
                    'total_costs': total_costs,
                    'gross_return_before_costs': 0,
                    'total_value': total_value
                })
                continue

            # Process subsequent dates
            previous_date = dates[i-1]
            
            # Get weights and cusips for previous and current portfolio
            previous_weights = results_df[results_df['date'] == previous_date]['weights'].iloc[0]
            previous_cusips = results_df[results_df['date'] == previous_date]['cusips'].iloc[0]
            new_weights = results_df[results_df['date'] == current_date]['weights'].iloc[0]
            new_cusips = results_df[results_df['date'] == current_date]['cusips'].iloc[0]

            # Calculate returns for the period
            weekly_return = self.get_weekly_returns(previous_date, current_date, 
                                                  list(set(previous_cusips) | set(new_cusips))).reset_index(names='cusip')
            
            # Calculate portfolio return before rebalancing
            previous_portfolio = pd.DataFrame({'cusip': previous_cusips, 'weight': previous_weights})
            previous_portfolio = pd.merge(previous_portfolio, weekly_return, on='cusip', how='left')
            previous_portfolio['Current_Return'] = previous_portfolio['Return'].fillna(0)
            portfolio_return = np.sum(previous_portfolio['weight'] * previous_portfolio['Current_Return'])
            
            # Update total portfolio value
            total_value *= (1 + portfolio_return)

            # Calculate annualization factor for costs
            days_between_rebalance = (current_date - previous_date).days
            annualization_factor = days_between_rebalance / 252
            
            # Get transaction costs
            transaction_costs = results_df[results_df['date'] == current_date]['transaction_costs'].iloc[0]
            
            # Combine all portfolio information
            all_cusips = list(set(previous_cusips) | set(new_cusips))
            all_weights = pd.DataFrame({
                'cusip': all_cusips,
                'previous_weight': [previous_weights[previous_cusips.index(cusip)] if cusip in previous_cusips else 0 for cusip in all_cusips],
                'new_weight': [new_weights[new_cusips.index(cusip)] if cusip in new_cusips else 0 for cusip in all_cusips]
            })
            
            # Add transaction costs and calculate weight changes
            all_weights = pd.merge(all_weights, transaction_costs, on='cusip', how='left')
            all_weights['weight_change'] = all_weights['new_weight'] - all_weights['previous_weight']
            
            # Add returns data
            all_weights = pd.merge(all_weights, weekly_return, on='cusip', how='left')
            all_weights['Return'] = all_weights['Return'].fillna(0)

            # Calculate short costs
            all_weights['short_cost'] = np.where(
                all_weights['new_weight'] < 0,
                (all_weights['short_cost'] / 10000) * np.abs(all_weights['weight_change']) * annualization_factor,
                0)
            
            # Calculate transaction costs
            all_weights['transaction_cost'] = (all_weights['trade_cost'] / 10000) * np.abs(all_weights['weight_change'])

            # Sum up total costs
            total_transaction_cost = all_weights['transaction_cost'].sum()
            total_short_cost = all_weights['short_cost'].sum() 
            total_costs = total_transaction_cost + total_short_cost
            gross_return_before_costs = portfolio_return

            # Update portfolio value after costs
            total_value -= total_value * (total_transaction_cost + total_short_cost)

            # Record period results
            portfolio_returns.append({
                'start_date': previous_date,
                'end_date': current_date,
                'portfolio_return_after_costs': portfolio_return - total_costs,
                'total_transaction_cost': -1*total_transaction_cost,
                'total_short_cost': -1*total_short_cost,
                'total_costs': total_costs,
                'gross_return_before_costs': gross_return_before_costs,
                'total_value': total_value
            })

        return pd.DataFrame(portfolio_returns)

    def run_backtest(self):
        """
        Executes the backtest over the specified date range.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing complete backtest results including portfolio returns and costs
        """
        # Get all relevant dates for the backtest
        alpha_dates = [date for date in self.data_retriever.get_alpha_dates() 
                      if self.start_date <= date <= self.end_date]
        rsqrm_dates = self.data_retriever.get_rsqrm_betas_dates()
        investable_universe_dates = self.data_retriever.get_investable_universe_dates()
        returns_dates = self.data_retriever.get_returns_dates()
        factor_dates = self.data_retriever.get_rsqrm_factor_dates()

        # Process each date in the backtest period
        for date in alpha_dates:
            print(f"Processing date: {date}")
            result = self.process_date(date, rsqrm_dates, factor_dates, 
                                    investable_universe_dates, returns_dates)
            self.results.append(result)

        # Create results DataFrame and calculate portfolio returns
        results_df = pd.DataFrame(self.results).dropna(how='all', axis=0)
        portfolio_returns = self.calculate_portfolio_returns(results_df)
        
        # Merge results with portfolio returns
        return pd.merge(results_df, portfolio_returns, 
                       left_on='date', right_on='end_date', how='left')

    def process_date(self, date, rsqrm_dates, factor_dates, investable_universe_dates, returns_dates):
        """
        Processes a single date in the backtest, including portfolio optimization.
        
        Parameters
        ----------
        date : datetime
            Current processing date
        rsqrm_dates : list
            Available dates for risk model data
        factor_dates : list
            Available dates for factor data
        investable_universe_dates : list
            Available dates for universe data
        returns_dates : list
            Available dates for returns data
            
        Returns
        -------
        dict
            Dictionary containing portfolio state and optimization results
        """
        # Load all required data for the current date
        alphas_df = self.data_retriever.load_alphas(date).dropna()
        risk_betas_df, risk_betas_date = self.get_most_recent_data(
            date, rsqrm_dates, self.data_retriever.load_risk_betas)
        investable_universe_df, inv_univ_date = self.get_most_recent_data(
            date, investable_universe_dates, self.data_retriever.load_investable_universe)
        returns_df = self.update_returns_data(date, returns_dates)
        risk_factors_df, risk_factors_date = self.get_most_recent_data(
            date, factor_dates, self.data_retriever.load_risk_factors)
        risk_correlations_df, risk_correlations_date = self.get_most_recent_data(
            date, factor_dates, self.data_retriever.load_risk_correlations)
        
        # Get investable universe
        universe_cusips = investable_universe_df['cusip'].tolist()
        
        # Build selection grid and select assets
        try:
            grid_builder = AlphaReturnGridBuilder(
                returns_df, universe_cusips, 
                n_days_returns=self.n_day_returns, 
                n_bins=self.alpha_bins
            )
            alpha_returns_bins = grid_builder.build_alpha_returns_grid(alphas_df)
        except Exception as e:
            print(f"Error building selection grid: {str(e)}")
            # Return previous portfolio state if grid building fails
            if self.last_portfolio is not None and len(self.last_portfolio) > 0:
                last_cusips = self.last_portfolio['cusip'].tolist()
                current_universe = set(universe_cusips)
                
                # Handle assets that left the universe
                removed_indices = [i for i, cusip in enumerate(last_cusips) 
                                 if cusip not in current_universe]
                
                if removed_indices:
                    # Adjust weights for removed assets
                    new_weights = np.copy(self.last_opt_weights)
                    new_weights[removed_indices] = 0
                    new_weights = self.normalize_weights(new_weights)
                    self.last_opt_weights = new_weights
            
            # Return current portfolio state
            return {
                'date': date,
                'risk_betas_date': risk_betas_date,
                'investable_universe_date': inv_univ_date,
                'returns_date_range': (returns_df.index.get_level_values('Date').min(), 
                                     returns_df.index.get_level_values('Date').max()),
                'num_assets': len(self.last_portfolio),
                'weights': self.last_opt_weights,
                'cusips': self.last_portfolio['cusip'].tolist(),
                'transaction_costs': self.last_transaction_costs,
                'portfolio_names': self.last_portfolio_names,
                'stock_betas': self.last_betas,
            }
            
        # Select assets using the grid
        asset_selector = AssetSelector(alpha_returns_bins)
        selected_assets_df = asset_selector.select_assets(
            corner_percent=30, 
            max_distance=5, 
            scale_factor=0.8, 
            taper_factor=2.5
        )

        # Calculate risk matrices
        correlation_matrix = risk_correlations_df.values
        std_matrix = np.diag(np.array(np.sqrt(risk_factors_df['Factor Variance'])))
        covariance_matrix = std_matrix @ correlation_matrix @ std_matrix

        # Process selected assets
        selected_long = selected_assets_df[selected_assets_df['corner'] == 'top_right']
        selected_short = selected_assets_df[selected_assets_df['corner'] == 'bottom_left']

        # Find new positions
        new_long = self.set_diff(selected_long['cusip'], self.active_long).reset_index(drop=True)
        new_short = self.set_diff(selected_short['cusip'], self.active_short).reset_index(drop=True)

        # Combine all relevant assets
        portfolio_cusips = self.set_union(
            self.set_union(new_long, new_short), 
            self.set_union(self.active_long, self.active_short)
        )

        # Prepare data for optimization
        betas_df = risk_betas_df.reset_index()[['cusip', 'betas']]
        betas_df['cusip'] = betas_df['cusip'].astype(str)
        betas_df = betas_df[betas_df['cusip'].isin(portfolio_cusips)]
        alphas_df = alphas_df[alphas_df['cusip'].isin(portfolio_cusips)]
        alphas_betas_df = pd.merge(
            alphas_df[['cusip', 'Alpha']], 
            betas_df[['cusip', 'betas']], 
            on='cusip', how='inner')

                # [continuing process_date method...]
        
        # Prepare portfolio universe
        portfolio_universe = alphas_betas_df[['cusip']].reset_index(drop=True)
        portfolio_universe = portfolio_universe[portfolio_universe['cusip'].isin(investable_universe_df['cusip'])]

        # Prepare risk betas
        risk_betas_df = risk_betas_df.reset_index()
        risk_betas_df = risk_betas_df[risk_betas_df['cusip'].isin(portfolio_universe['cusip'])].sort_values('cusip')
        risk_betas_df = risk_betas_df.reset_index(drop=True)
        risk_betas_expanded = expand_betas(risk_betas_df)

        # Initialize optimization parameters
        if self.opt_count == 0:
            # First optimization
            w_star_t = np.zeros(len(portfolio_universe))
            lambda_l1 = 0
        else:
            # Subsequent optimizations
            self.active_long = self.active_long[self.active_long.isin(investable_universe_df['cusip'])]
            self.active_short = self.active_short[self.active_short.isin(investable_universe_df['cusip'])]
            lambda_l1 = 0.25
            
            # Prepare new portfolio with previous weights
            new_portfolio = pd.DataFrame({
                'cusip': portfolio_universe['cusip'], 
                'weights': np.zeros(len(portfolio_universe['cusip']))
            })
            new_portfolio.set_index('cusip', inplace=True)
            self.last_portfolio.set_index('cusip', inplace=True)
            
            # Transfer previous weights to new portfolio
            common_cusips = new_portfolio.index.intersection(self.last_portfolio.index)
            new_portfolio.loc[common_cusips, 'weights'] = self.last_portfolio.loc[common_cusips, 'weights']
            new_portfolio.reset_index(inplace=True)
            portfolio_universe = new_portfolio[['cusip']]
            w_star_t = new_portfolio['weights'].to_numpy()

        # Run portfolio optimization
        w, w_plus, w_minus = self.optimize_portfolio_jacobian(
            self.active_long, self.active_short, new_long, new_short,
            portfolio_universe, risk_betas_expanded.to_numpy(),
            risk_betas_df['monthly_residual_sd_decimal'].to_numpy(),
            w_star_t, covariance_matrix, lambda_l1,
            min_weight=0.00001,
            factor_weight=0.8
        )
        
        # Post-optimization processing
        optimal_weights = self.remove_small_weights(w)
        optimal_weights = self.normalize_weights(optimal_weights)

        # Calculate and print factor exposures for monitoring
        factor_exposures = np.dot(risk_betas_expanded.to_numpy()[:, :37].T, optimal_weights)
        print("\nFactor Exposures:")
        print(f"Max absolute exposure: {np.abs(factor_exposures).max():.6f}")
        print(f"Mean absolute exposure: {np.abs(factor_exposures).mean():.6f}")
        print(f"RMS exposure: {np.sqrt(np.mean(factor_exposures**2)):.6f}")
        
        # Print significant factor exposures
        print("\nPer-factor exposures:")
        for i, exposure in enumerate(factor_exposures):
            if abs(exposure) > 0.01:
                print(f"Factor {i}: {exposure:.6f}")

        # Update portfolio state
        self.last_opt_weights = optimal_weights
        self.last_portfolio = pd.DataFrame({
            'cusip': portfolio_universe['cusip'], 
            'weights': optimal_weights
        })
        
        # Remove zero-weight assets from tracking
        zero_assets = self.last_portfolio[self.last_portfolio['weights'] == 0]['cusip'].tolist()
        self.active_long = [x for x in self.active_long if x not in zero_assets]
        self.active_short = [x for x in self.active_short if x not in zero_assets]

        # Update optimization counter and active positions
        self.opt_count += 1
        self.active_long = self.set_union(self.active_long, new_long).reset_index(drop=True)
        self.active_short = self.set_union(self.active_short, new_short).reset_index(drop=True)

        # Calculate transaction costs
        universe_caps = alphas_df[alphas_df['cusip'].isin(investable_universe_df['cusip'])][
            ['cusip', 'Trade.Cost.Mkt.Cap', 'Alpha']]
        max_universe_cap = universe_caps['Trade.Cost.Mkt.Cap'].max()
        min_universe_cap = universe_caps['Trade.Cost.Mkt.Cap'].min()
        transaction_costs = self.get_transaction_short_costs(
            universe_caps, max_universe_cap, min_universe_cap)

        # Update portfolio information
        self.last_transaction_costs = transaction_costs
        portfolio_names = investable_universe_df[
            investable_universe_df['cusip'].isin(selected_assets_df['cusip'])][['cusip','Name']]
        portfolio_names = portfolio_names.set_index('cusip').reindex(portfolio_universe['cusip'])['Name'].tolist()
        self.last_portfolio_names = portfolio_names
        self.last_betas = alphas_betas_df['betas'].tolist()

        # Return results
        return {
            'date': date,
            'risk_betas_date': risk_betas_date,
            'investable_universe_date': inv_univ_date,
            'returns_date_range': (
                returns_df.index.get_level_values('Date').min(), 
                returns_df.index.get_level_values('Date').max()
            ),
            'num_assets': len(optimal_weights),
            'weights': optimal_weights,
            'cusips': portfolio_universe['cusip'].tolist(),
            'transaction_costs': transaction_costs,
            'portfolio_names': portfolio_names,
            'stock_betas': alphas_betas_df['betas'].tolist()
        }

            def optimize_portfolio_jacobian(self, active_long, active_short, new_long, new_short, 
                                  portfolio_universe, betas, risk_residuals, w_star_t, 
                                  sigma_f, lambda_l1, min_weight=0.00001, factor_weight=0.2):
        """
        Optimizes portfolio weights using factor neutrality constraints and transaction costs.
        
        Parameters
        ----------
        active_long : pd.Series
            Currently active long positions
        active_short : pd.Series
            Currently active short positions
        new_long : pd.Series
            New long positions to be added
        new_short : pd.Series
            New short positions to be added
        portfolio_universe : pd.DataFrame
            Universe of available assets
        betas : np.ndarray
            Factor betas for each asset
        risk_residuals : np.ndarray
            Residual risk for each asset
        w_star_t : np.ndarray
            Previous portfolio weights
        sigma_f : np.ndarray
            Factor covariance matrix
        lambda_l1 : float
            L1 regularization parameter
        min_weight : float, optional
            Minimum weight threshold, by default 0.00001
        factor_weight : float, optional
            Weight for factor neutrality objective, by default 0.2
            
        Returns
        -------
        tuple
            (optimal_weights, positive_weights, negative_weights)
        """
        # Determine position categories
        invested_new_longs = self.set_diff(new_long, active_short)
        invested_new_shorts = self.set_diff(new_short, active_long)
        re_active_longs = self.set_diff(active_long, self.set_union(new_long, new_short))
        re_active_shorts = self.set_diff(active_short, self.set_union(new_long, new_short))
        direction_changes = self.set_union(
            self.set_intersect(new_long, active_short), 
            self.set_intersect(new_short, active_long)
        )

        num_assets = len(portfolio_universe)
        total_vars = 3 * num_assets

        # Add regularization to prevent numerical instability
        epsilon = 1e-5
        sigma_t = np.matmul(np.matmul(betas, sigma_f), betas.T) + np.diag(risk_residuals**2) + epsilon * np.eye(num_assets)

        def objective(w):
            """Objective function combining risk and factor neutrality."""
            w_original = w[:num_assets]
            w_plus = w[num_assets:2*num_assets]
            w_minus = w[2*num_assets:]
            
            # Portfolio risk objective
            original_obj = 0.5 * np.dot(w_original.T, np.dot(sigma_t, w_original)) + \
                         lambda_l1 * (np.sum(w_plus) + np.sum(w_minus))
            
            # Factor neutrality objective (per-factor)
            factor_exposures = np.dot(betas[:, :37].T, w_original)
            factor_obj = np.sum(factor_exposures**2)
            
            return (1 - factor_weight) * original_obj + factor_weight * factor_obj

        def objective_jacobian(w):
            """Jacobian of the objective function."""
            w_original = w[:num_assets]
            
            # Original objective gradient
            original_grad_w = np.dot(sigma_t, w_original)
            original_grad = np.concatenate([
                original_grad_w,
                lambda_l1 * np.ones(num_assets),
                lambda_l1 * np.ones(num_assets)
            ])
            
            # Factor neutrality gradient
            factor_exposures = np.dot(betas[:, :37].T, w_original)
            factor_grad_w = 2 * np.dot(betas[:, :37], factor_exposures)
            factor_grad = np.concatenate([
                factor_grad_w,
                np.zeros(2 * num_assets)
            ])
            
            return (1 - factor_weight) * original_grad + factor_weight * factor_grad

        # Define constraint Jacobians
        def jac_sum_zero(w):
            """Jacobian for sum of weights constraint."""
            return np.hstack([np.ones(num_assets), np.zeros(2*num_assets)])

        def jac_sum_pos_one(w):
            """Jacobian for sum of positive weights = 1 constraint."""
            jac = np.zeros(3*num_assets)
            jac[:num_assets] = (w[:num_assets] > 0).astype(float)
            return jac

        def jac_w_star_t(w):
            """Jacobian for weight decomposition constraint."""
            jac = np.zeros((num_assets, 3*num_assets))
            jac[:, :num_assets] = np.eye(num_assets)
            jac[:, num_assets:2*num_assets] = np.eye(num_assets)
            jac[:, 2*num_assets:] = -np.eye(num_assets)
            return jac

        def jac_ineq(w, i):
            """Jacobian for inequality constraints."""
            jac = np.zeros(3*num_assets)
            jac[i] = 1
            return jac

        def jac_ineq_neg(w, i):
            """Jacobian for negative inequality constraints."""
            jac = np.zeros(3*num_assets)
            jac[i] = -1
            return jac

        def constraint_min_weight(w, i):
            """Minimum weight constraint."""
            weight = w[i]
            return (weight >= min_weight) | (weight <= -min_weight) | (weight == 0)

        def jac_min_weight(w, i):
            """Jacobian for minimum weight constraint."""
            jac = np.zeros(total_vars)
            if w[i] > 0:
                jac[i] = 1
            elif w[i] < 0:
                jac[i] = -1
            return jac

        # Define optimization constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w[:num_assets]), 'jac': jac_sum_zero},
            {'type': 'eq', 'fun': lambda w: np.sum(np.maximum(w[:num_assets], 0)) - 1, 'jac': jac_sum_pos_one},
            {'type': 'eq', 'fun': lambda w: w[:num_assets] + w[num_assets:2*num_assets] - w[2*num_assets:] - w_star_t, 
             'jac': jac_w_star_t}
        ]

        # Add position-specific constraints
        for i, cusip in enumerate(portfolio_universe['cusip']):
            # Minimum weight constraint
            constraints.append({
                'type': 'ineq', 
                'fun': lambda w, i=i: constraint_min_weight(w, i), 
                'jac': lambda w, i=i: jac_min_weight(w, i)
            })
            
            # Position-specific constraints
            if cusip in invested_new_longs:
                constraints.append({'type': 'ineq', 'fun': lambda w, i=i: w[i], 'jac': lambda w, i=i: jac_ineq(w, i)})
                constraints.append({'type': 'ineq', 'fun': lambda w, i=i: 1 - w[i], 'jac': lambda w, i=i: jac_ineq_neg(w, i)})
            elif cusip in invested_new_shorts:
                constraints.append({'type': 'ineq', 'fun': lambda w, i=i: -w[i], 'jac': lambda w, i=i: jac_ineq_neg(w, i)})
                constraints.append({'type': 'ineq', 'fun': lambda w, i=i: w[i] + 1, 'jac': lambda w, i=i: jac_ineq(w, i)})
            elif cusip in re_active_longs:
                constraints.append({'type': 'ineq', 'fun': lambda w, i=i: w[i], 'jac': lambda w, i=i: jac_ineq(w, i)})
                constraints.append({'type': 'ineq', 'fun': lambda w, i=i: w_star_t[i] - w[i], 'jac': lambda w, i=i: jac_ineq_neg(w, i)})
            elif cusip in re_active_shorts:
                constraints.append({'type': 'ineq', 'fun': lambda w, i=i: -w[i], 'jac': lambda w, i=i: jac_ineq_neg(w, i)})
                constraints.append({'type': 'ineq', 'fun': lambda w, i=i: w[i] - w_star_t[i], 'jac': lambda w, i=i: jac_ineq(w, i)})
            elif cusip in direction_changes:
                constraints.append({'type': 'eq', 'fun': lambda w, i=i: w[i], 'jac': lambda w, i=i: jac_ineq(w, i)})

        # Define bounds
        bounds = [(None, None) for _ in range(num_assets)] + [(0, None) for _ in range(2*num_assets)]

        # Initialize optimization
        x0 = np.concatenate([self.create_initial_guess(num_assets), np.zeros(2*num_assets)])

        # Run optimization
        result = minimize(objective, x0, method='SLSQP', jac=objective_jacobian, 
                        constraints=constraints, bounds=bounds, 
                        options={'maxiter': 1000, 'disp': True, 'ftol': 1e-8, 'eps': 1e-10})

        if not result.success:
            raise ValueError(f"Optimization failed. Status: {result.message}")

        return result.x[:num_assets], result.x[num_assets:2*num_assets], result.x[2*num_assets:]

    @staticmethod
    def set_diff(set1, set2):
        """
        Computes the set difference between two sets.
        
        Parameters
        ----------
        set1 : pd.Series or list
            First set of elements
        set2 : pd.Series or list
            Second set of elements
            
        Returns
        -------
        pd.Series
            Elements in set1 that are not in set2
        """
        return pd.Series(list(set(set1) - set(set2)))

    @staticmethod
    def set_union(set1, set2):
        """
        Computes the union of two sets.
        
        Parameters
        ----------
        set1 : pd.Series or list
            First set of elements
        set2 : pd.Series or list
            Second set of elements
            
        Returns
        -------
        pd.Series
            Union of set1 and set2
        """
        return pd.Series(list(set(set1) | set(set2)))

    @staticmethod
    def set_intersect(set1, set2):
        """
        Computes the intersection of two sets.
        
        Parameters
        ----------
        set1 : pd.Series or list
            First set of elements
        set2 : pd.Series or list
            Second set of elements
            
        Returns
        -------
        pd.Series
            Intersection of set1 and set2
        """
        return pd.Series(list(set(set1) & set(set2)))

    @staticmethod
    def create_initial_guess(num_assets):
        """
        Creates an initial guess for portfolio optimization.
        
        Parameters
        ----------
        num_assets : int
            Number of assets in the portfolio
            
        Returns
        -------
        np.ndarray
            Initial weight guess for each asset
        """
        return np.random.uniform(-0.5, 0.5, num_assets)

    @staticmethod
    def normalize_weights(weights):
        """
        Normalizes portfolio weights to ensure long and short sides sum to 1 and -1.
        
        Parameters
        ----------
        weights : np.ndarray
            Array of portfolio weights
            
        Returns
        -------
        np.ndarray
            Normalized weights
        
        Notes
        -----
        This method ensures that the sum of positive weights equals 1 and the sum of
        negative weights equals -1, maintaining a dollar-neutral portfolio.
        """
        positive_sum = np.sum(np.maximum(weights, 0))
        negative_sum = np.sum(np.minimum(weights, 0))
        return np.where(weights > 0, 
                       weights / positive_sum, 
                       weights / abs(negative_sum))
    
    @staticmethod
    def remove_small_weights(weights, min_weight=0.00001):
        """
        Removes weights below a minimum threshold.
        
        Parameters
        ----------
        weights : np.ndarray
            Array of portfolio weights
        min_weight : float, optional
            Minimum absolute weight threshold, by default 0.00001
            
        Returns
        -------
        np.ndarray
            Weights with small positions removed
            
        Notes
        -----
        This method helps reduce portfolio complexity by eliminating very small positions
        that may not be practically meaningful.
        """
        return np.where(abs(weights) < min_weight, 0, weights)

    def get_sigma_r(self, betas, risk_residuals, sigma_f):
        """
        Calculates the total risk matrix including systematic and idiosyncratic risk.
        
        Parameters
        ----------
        betas : np.ndarray
            Factor betas for each asset
        risk_residuals : np.ndarray
            Residual risk for each asset
        sigma_f : np.ndarray
            Factor covariance matrix
            
        Returns
        -------
        np.ndarray
            Total risk matrix
            
        Notes
        -----
        This method combines systematic risk (factor exposures) and idiosyncratic risk
        to create a complete risk model for portfolio optimization.
        """
        return np.dot(np.dot(betas, sigma_f), betas.T) + np.diag(risk_residuals**2)