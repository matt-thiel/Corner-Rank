import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class AlphaReturnGridBuilder:
    """
    A class for building and analyzing alpha-return grids.

    This class processes returns data, alpha data, and an investable universe to create
    a grid representation of alpha and return relationships.

    Parameters
    ----------
    returns_df : pandas.DataFrame
        DataFrame containing returns data.
    universe_cusips : array_like
        List or array of CUSIPs representing the investable universe.
    n_days_returns : int, optional
        Number of days to use for calculating returns (default is 7).
    n_bins : int, optional
        Number of bins to use for alpha and return percentiles (default is 20).

    Attributes
    ----------
    n_days_returns : int
        Number of days used for calculating returns.
    n_bins : int
        Number of bins used for alpha and return percentiles.
    investable_universe_df : pandas.DataFrame
        DataFrame representing the investable universe.
    alpha_universe_df : pandas.DataFrame
        DataFrame containing alpha data for the investable universe.
    close_prices : pandas.DataFrame
        DataFrame containing close prices (not used in current implementation).
    daily_returns_df : pandas.DataFrame
        DataFrame containing daily returns data.
    returns_df : pandas.DataFrame
        DataFrame containing processed returns data.
    alpha_returns_grid_df : pandas.DataFrame
        DataFrame representing the alpha-returns grid.
    grid_df : pandas.DataFrame
        DataFrame representing the binned grid data.
    universe : array_like
        List or array of CUSIPs representing the investable universe.
    """

    def __init__(self, returns_df, universe_cusips, n_days_returns=7, n_bins=20):
        self.n_days_returns = n_days_returns
        self.n_bins = n_bins
        self.investable_universe_df = pd.DataFrame()
        self.alpha_universe_df = pd.DataFrame()
        self.close_prices = pd.DataFrame()
        self.daily_returns_df = returns_df.groupby('cusip').tail(n_days_returns)
        self.returns_df = pd.DataFrame()
        self.alpha_returns_grid_df = pd.DataFrame()
        self.grid_df = pd.DataFrame()
        self.universe = universe_cusips
        
    def get_investable_universe(self):
        """
        Create a DataFrame of the investable universe.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the investable universe CUSIPs.
        """
        self.investable_universe_df = pd.DataFrame(self.universe, columns=['cusip'])
        return self.investable_universe_df

    def get_alpha(self, alphas):
        """
        Process the alpha data and filter it to the investable universe.

        Parameters
        ----------
        alphas : pandas.DataFrame
            DataFrame containing alpha data.

        Returns
        -------
        pandas.DataFrame
            Filtered alpha data for the investable universe.
        """
        if self.investable_universe_df.empty:
            self.get_investable_universe()
        
        self.alpha_universe_df = alphas[alphas.cusip.isin(self.investable_universe_df.cusip)].sort_values('cusip').reset_index(drop=True)
        return self.alpha_universe_df

    def get_returns(self):
        """
        Calculate the total return over the supplied days for each CUSIP.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by CUSIP with date and total_return columns.
        """
        returns = self.daily_returns_df.reset_index()
        
        total_returns = returns.groupby('cusip').agg({
            'Return': lambda x: (1 + x).prod() - 1,
            'Date': 'last'
        })
        
        return total_returns.sort_index()

    def build_alpha_returns_grid(self, alphas):
        """
        Build the alpha-returns grid by merging alpha and returns data.

        Parameters
        ----------
        alphas : pandas.DataFrame
            DataFrame containing alpha data.

        Returns
        -------
        pandas.DataFrame
            DataFrame representing the alpha-returns grid.
        """
        if self.alpha_universe_df.empty:
            self.get_alpha(alphas)

        self.alpha_universe_df['cusip'] = self.alpha_universe_df['cusip'].astype(str)
        self.returns_df = self.get_returns().reset_index()
        self.returns_df['cusip'] = self.returns_df['cusip'].astype(str)
        self.alpha_returns_grid_df = pd.merge(self.alpha_universe_df, self.returns_df, on='cusip', how='inner')
        self.alpha_returns_grid_df['alpha_percentile'] = self.alpha_returns_grid_df['Alpha'].rank(pct=True)
        self.alpha_returns_grid_df['return_percentile'] = self.alpha_returns_grid_df['Return'].rank(pct=True)
        self.alpha_returns_grid_df['alpha_bin'] = pd.cut(self.alpha_returns_grid_df['alpha_percentile'], bins=self.n_bins, labels=False)
        self.alpha_returns_grid_df['return_bin'] = pd.cut(self.alpha_returns_grid_df['return_percentile'], bins=self.n_bins, labels=False)
        
        return self.alpha_returns_grid_df

    def plot_grid(self):
        """
        Plot a heatmap of the alpha-returns grid.

        This method creates a heatmap visualization of the binned alpha-returns data.
        """
        if self.alpha_returns_grid_df.empty:
            self.build_alpha_returns_grid()
        
        self.grid_df = self.alpha_returns_grid_df.groupby(['return_bin', 'alpha_bin']).size().unstack(fill_value=0)

        plt.figure(figsize=(8, 6))
        sns.heatmap(self.grid_df.iloc[::-1], annot=True, fmt="d", cmap="Blues")
        plt.title('Heatmap of Points in Bins')
        plt.show()
        

class AssetSelector:
    """
    A class for selecting assets based on alpha and return characteristics.

    This class bins alpha and return data, calculates selection weights,
    and selects assets based on specified criteria.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing alpha and return data for assets.
    n_bins : int, optional
        Number of bins to use for alpha and return data (default is 10).

    Attributes
    ----------
    df : pandas.DataFrame
        Input DataFrame containing asset data.
    n_bins : int
        Number of bins used for alpha and return data.
    """

    def __init__(self, df, n_bins=10):
        self.df = df
        self.n_bins = n_bins

    def bin_data(self):
        """
        Bin the alpha and return data into specified number of bins.
        """
        self.df['alpha_bin'] = pd.qcut(self.df['Alpha'], q=self.n_bins, labels=False)
        self.df['return_bin'] = pd.qcut(self.df['Return'], q=self.n_bins, labels=False)

    def selection_weight(self, alpha, return_, max_distance, scale_factor, taper_factor):
        """
        Calculate the selection weight for an asset based on its alpha and return bins.

        Parameters
        ----------
        alpha : int
            Alpha bin of the asset.
        return_ : int
            Return bin of the asset.
        max_distance : float
            Maximum allowed distance from the diagonal.
        scale_factor : float
            Factor to scale the maximum allowed distance.
        taper_factor : float
            Factor to control tapering of the selection area.

        Returns
        -------
        float
            Selection weight of the asset.
        """
        center = self.n_bins / 2
        dist_from_center = max(abs(alpha - center), abs(return_ - center))
        relative_dist = dist_from_center / center
        taper = np.power(relative_dist, taper_factor)
        max_allowed_distance = max_distance * (taper + (1 - taper) * 0.2) * scale_factor
        diagonal_dist = abs(alpha - return_) / np.sqrt(2)
        
        if diagonal_dist <= max_allowed_distance:
            diagonal_weight = (max_allowed_distance - diagonal_dist) / max_allowed_distance
            center_weight = relative_dist
            return diagonal_weight * center_weight
        return 0

    def select_assets(self, corner_percent=20, max_distance=5, scale_factor=0.5, taper_factor=1.0):
        """
        Select assets based on specified criteria.

        Parameters
        ----------
        corner_percent : float, optional
            Percentage of assets to select from each corner (default is 20).
        max_distance : float, optional
            Maximum allowed distance from the diagonal (default is 5).
        scale_factor : float, optional
            Factor to scale the maximum allowed distance (default is 0.5).
        taper_factor : float, optional
            Factor to control tapering of the selection area (default is 1.0).

        Returns
        -------
        pandas.DataFrame
            DataFrame containing selected assets and their characteristics.
        """
        self.df['selection_weight'] = self.df.apply(
            lambda x: self.selection_weight(x['alpha_bin'], x['return_bin'], max_distance, scale_factor, taper_factor), 
            axis=1
        )
        
        selected_area = self.df[self.df['selection_weight'] > 0]
        top_right = selected_area[(selected_area['alpha_bin'] >= self.n_bins / 2) & (selected_area['return_bin'] >= self.n_bins / 2)]
        bottom_left = selected_area[(selected_area['alpha_bin'] < self.n_bins / 2) & (selected_area['return_bin'] < self.n_bins / 2)]
        
        assets_per_corner = int(len(selected_area) * corner_percent / 100 / 2)
        
        top_right_selected = top_right.nlargest(assets_per_corner, 'selection_weight')
        bottom_left_selected = bottom_left.nlargest(assets_per_corner, 'selection_weight')
        
        top_right_selected['corner'] = 'top_right'
        bottom_left_selected['corner'] = 'bottom_left'
        
        selected = pd.concat([top_right_selected, bottom_left_selected])
        selected['corner_percent'] = corner_percent
        
        return selected

    def plot_selection(self, selected):
        """
        Plot the selected assets on a scatter plot.

        Parameters
        ----------
        selected : pandas.DataFrame
            DataFrame containing the selected assets.
        """
        plt.figure(figsize=(12, 10))
        sns.scatterplot(data=self.df, x='Alpha', y='Return', alpha=0.3)
        sns.scatterplot(data=selected, x='Alpha', y='Return', hue='corner', style='corner', s=100)
        plt.title('Selected Assets')
        plt.xlabel('Alpha')
        plt.ylabel('Return')
        plt.legend(title='Corner')
        plt.show()

    def plot_selection_area(self, max_distance=5, scale_factor=0.5, taper_factor=1.0):
        """
        Plot the selection area as a heatmap.

        Parameters
        ----------
        max_distance : float, optional
            Maximum allowed distance from the diagonal (default is 5).
        scale_factor : float, optional
            Factor to scale the maximum allowed distance (default is 0.5).
        taper_factor : float, optional
            Factor to control tapering of the selection area (default is 1.0).
        """
        x = np.arange(self.n_bins)
        y = np.arange(self.n_bins)
        X, Y = np.meshgrid(x, y)
        
        selection_area = np.vectorize(lambda a, b: self.selection_weight(a, b, max_distance, scale_factor, taper_factor))(X, Y)

        plt.figure(figsize=(12, 10))
        plt.imshow(selection_area, cmap='Blues', alpha=0.3, vmin=0, vmax=selection_area.max())
        plt.title(f'Selection Area (max_distance: {max_distance}, scale: {scale_factor}, taper: {taper_factor})')
        plt.xlabel('Alpha Bin')
        plt.ylabel('Return Bin')
        plt.gca().invert_yaxis()
        plt.colorbar(label='Selection Weight')
        plt.show()