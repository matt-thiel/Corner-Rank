from cornerRank import CornerRank
from datetime import datetime
import pandas as pd
from formatResultsOOS import format_betas_for_excel, format_portfolio_performance, format_backtest_weights

def main():
    # Set up backtest parameters
    #start_date = '2017-12-27'
    #end_date = '2024-07-31'
    #start_date = '2015-08-13'
    #end_date = '2016-07-31'
    #start_date = '2021-01-01'
    #start_date = '2017-06-01'
    # OOS
    start_date = '2014-08-13'
    #start_date = '2015-08-10'
    #end_date = '2016-05-01'
    #end_date = '2017-02-02'
    #start_date = '2021-01-01'
    #end_date = '2016-06-09'
    end_date = '2021-06-09'
    # IS
    #start_date = '2017-12-27'
    #end_date = '2024-07-31'

    # current test is with alpha replacement not weight replacement
    #good results will fill forward weights
    
    n_day_returns = 7
    alpha_bins = 30
    max_returns_days = 15

    print(f"Starting backtest from {start_date} to {end_date}")

    # Initialize and run the backtester
    backtester = CornerRank(
        start_date=start_date,
        end_date=end_date,
        n_day_returns=n_day_returns,
        alpha_bins=alpha_bins,
        max_returns_days=max_returns_days
    )

    results_df = backtester.run_backtest()

    # Generate a unique filename for the results
    results_filename = f"backtest_results_{start_date}_to_{end_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    results_path = f"results/{results_filename}"

    # Save the results to a CSV file
    results_df.to_csv(results_path, index=False)
    print(f"Backtest results saved to: {results_path}")

    # Print summary statistics
    print("\nSummary Statistics:")
    print(results_df[['end_date', 'total_value']])

    print('Formatting Weights...')
    format_backtest_weights()
    print('Finished Weights')

    print('Formatting Portfolio Betas...')
    portfolio_betas = format_betas_for_excel()
    print('Finished Portfolio Betas')

    print('Formatting Portfolio Performance...')
    format_portfolio_performance()
    print('Finished Portfolio Performance')




if __name__ == "__main__":
    main()

