import os



# Get the current working directory as the project directory
project_dir = os.getcwd()

# Define paths for different data directories
# In sample
# Directory for alpha data files
is_alpha_dir = os.path.join(project_dir, 'data', 'in_sample', 'Alphas') 
#raw_alpha_dir = os.path.join(project_dir, 'data', 'Alphas')
# Directory for risk model data files
is_rsqrm_dir = os.path.join(project_dir, 'data','in_sample', 'US 2_19_9g')
#rsqrm_oos_dir = os.path.join(project_dir, 'data', 'US 2_19_9a_OOS')
# Directory for investable universe data files
is_investable_universe_dir = os.path.join(project_dir, 'data','in_sample', 'InvestableUniverse')
#raw_investable_universe_dir = os.path.join(project_dir, 'data', 'InvestableUniverse')
# Directory for returns data files
is_returns_dir = os.path.join(project_dir, 'data','in_sample', 'Returns')

## Out of Sample
oos_alpha_dir = os.path.join(project_dir, 'data', 'out_of_sample', 'Alphas') 
oos_raw_alpha_dir = os.path.join(project_dir, 'data', 'out_of_sample', 'AlphasRaw')
# Directory for risk model data files
oos_rsqrm_dir = os.path.join(project_dir, 'data','out_of_sample', 'US 2_19_9g')
#oos_rsqrm_model_dir = os.path.join(project_dir, 'data','out_of_sample', 'US 2_19_9a_OOS')
# Directory for investable universe data files
oos_investable_universe_dir = os.path.join(project_dir, 'data','out_of_sample', 'InvestableUniverse')
oos_raw_investable_universe_dir = os.path.join(project_dir, 'data','out_of_sample', 'InvestableUniverseRaw')
# Directory for returns data files
oos_returns_dir = os.path.join(project_dir, 'data','out_of_sample', 'Returns')
oos_raw_returns_dir = os.path.join(project_dir, 'data','out_of_sample', 'ReturnsRaw')
oos_model_exposures_dir = os.path.join(project_dir, 'data','out_of_sample', 'model exposures')

# Full Sample
# Directory for alpha data files
full_alpha_dir = os.path.join(project_dir, 'data', 'full_sample', 'Alphas') 
#raw_alpha_dir = os.path.join(project_dir, 'data', 'Alphas')
# Directory for risk model data files
full_rsqrm_dir = os.path.join(project_dir, 'data','full_sample', 'US 2_19_9g')
#rsqrm_oos_dir = os.path.join(project_dir, 'data', 'US 2_19_9a_OOS')
# Directory for investable universe data files
full_investable_universe_dir = os.path.join(project_dir, 'data','full_sample', 'InvestableUniverse')
#raw_investable_universe_dir = os.path.join(project_dir, 'data', 'InvestableUniverse')
# Directory for returns data files
full_returns_dir = os.path.join(project_dir, 'data','full_sample', 'Returns')