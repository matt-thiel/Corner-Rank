import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def create_analysis_plots():
    # Find and load the latest formatted betas file
    results_folder = 'results'
    formatted_folder = 'formatted/model_exposure_performance'
    import os
    import glob
    
    files = glob.glob(os.path.join(results_folder, formatted_folder, 'formatted_rsqrm_betas_*.xlsx'))
    latest_file = max(files, key=os.path.getmtime)
    portfolio_betas = pd.read_excel(latest_file, sheet_name='Portfolio Betas', index_col=0)

    # 1. Heatmap of factor exposures over time
    plt.figure(figsize=(20, 12))
    sns.heatmap(portfolio_betas, cmap='RdBu', center=0, 
                vmin=-0.5, vmax=0.5, cbar_kws={'label': 'Factor Exposure'})
    plt.title('Factor Exposure Heatmap Over Time')
    plt.xlabel('Date')
    plt.ylabel('Factor')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('factor_exposure_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 2. Box plot of factor exposures with improved label formatting
    plt.figure(figsize=(20, 10))
    
    # Create box plot
    ax = portfolio_betas.T.boxplot(figsize=(20, 10))
    
    # Rotate and align the tick labels so they look better
    plt.xticks(range(1, len(portfolio_betas.index) + 1), portfolio_betas.index, 
               rotation=45, ha='right')
    
    # Use a tight layout, but adjust the bottom margin to prevent label cutoff
    plt.gcf().subplots_adjust(bottom=0.2)
    
    # Add title and labels
    plt.title('Distribution of Factor Exposures', pad=20)
    plt.ylabel('Exposure')
    plt.grid(True, alpha=0.3)
    
    # Save with extra space at the bottom for labels
    plt.savefig('factor_exposure_distributions.png', 
                bbox_inches='tight', 
                dpi=300)
    plt.close()

    # 3. Rolling volatility of factor exposures
    rolling_vol = portfolio_betas.rolling(window=20, axis=1).std()
    plt.figure(figsize=(15, 8))
    rolling_vol.T.plot(figsize=(15, 8), alpha=0.7)
    plt.title('20-Period Rolling Volatility of Factor Exposures')
    plt.xlabel('Date')
    plt.ylabel('Rolling Std Dev')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig('factor_exposure_volatility.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 4. Factor exposure magnitude over time
    abs_exposures = portfolio_betas.abs()
    total_exposure = abs_exposures.sum()
    plt.figure(figsize=(15, 8))
    plt.plot(total_exposure.index, total_exposure.values)
    plt.title('Total Absolute Factor Exposure Over Time')
    plt.xlabel('Date')
    plt.ylabel('Sum of Absolute Exposures')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('total_factor_exposure.png', bbox_inches='tight', dpi=300)
    plt.close()

    # 5. Style vs Sector exposure comparison
    style_factors = [
        "Dividend Yield", "Value", "Growth Trend", "Growth Momentum",
        "Short Term Price Momentum", "Long Term Price Momentum",
        "Leverage", "Liquidity", "Quality", "US Market Large", "US Market Small"
    ]
    
    sector_factors = [
        "Energy Equipment & Services", "Materials", "Aerospace & Defence",
        "Building & Construction", "Industrials", "Transport",
        "Consumer Discretionary", "Retailers", "Consumer Staples",
        "Health Care", "Biotechnology & Pharmaceuticals", "Banking",
        "Diversified Financials", "Capital Markets", "Insurance",
        "Real Estate", "Software & IT Services", "Hardware & Technology",
        "Telecom Services", "Utilities"
    ]

    style_exposure = abs_exposures.loc[style_factors].sum()
    sector_exposure = abs_exposures.loc[sector_factors].sum()

    plt.figure(figsize=(15, 8))
    plt.plot(style_exposure.index, style_exposure.values, label='Style Factors', linewidth=2)
    plt.plot(sector_exposure.index, sector_exposure.values, label='Sector Factors', linewidth=2)
    plt.title('Style vs Sector Total Absolute Exposure')
    plt.xlabel('Date')
    plt.ylabel('Sum of Absolute Exposures')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('style_vs_sector_exposure.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    create_analysis_plots()