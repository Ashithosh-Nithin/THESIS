#!/usr/bin/env python3
"""
Enrollment Forecasting Analysis Script - USING ACTUAL IPEDS DATA
==================================================================

This script performs all analyses for the thesis using the real IPEDS panel dataset.
Author: Ashithosh Nithin
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Configuration
RANDOM_STATE = 42
TRAIN_START = 2010
TRAIN_END = 2017
TEST_START = 2018
TEST_END = 2021
DATA_PATH = '/mnt/user-data/uploads/ipeds_panel_2010_2021.parquet'

np.random.seed(RANDOM_STATE)

#==============================================================================
# 1. DATA LOADING
#==============================================================================

def load_and_prepare_data():
    """Load actual IPEDS panel data"""
    print("=" * 80)
    print("LOADING ACTUAL IPEDS DATA")
    print("=" * 80)
    
    df = pd.read_parquet(DATA_PATH)
    
    # Rename for consistency
    df = df.rename(columns={
        'unitid': 'institution_id',
        'adm_number_enrolled_total': 'enrollment',
        'adm_number_applied': 'applications',
        'adm_number_admitted': 'admissions',
        'adm_rate_acceptance': 'acceptance_rate',
        'adm_rate_yield_fulltime': 'yield_rate',
        'sfa_netprice_avg': 'net_price',
        'sfa_grantsandscholarships_avg': 'grant_aid',
        'sfr_ratio': 'student_faculty_ratio'
    })
    
    # Calculate rates if needed
    if 'acceptance_rate' not in df.columns and 'applications' in df.columns and 'admissions' in df.columns:
        df['acceptance_rate'] = df['admissions'] / df['applications']
    if 'yield_rate' not in df.columns and 'admissions' in df.columns and 'enrollment' in df.columns:
        df['yield_rate'] = df['enrollment'] / df['admissions']
    
    print(f"✓ Loaded {len(df)} institution-year observations")
    print(f"✓ Years: {df['year'].min()}-{df['year'].max()}")
    print(f"✓ Institutions: {df['institution_id'].nunique()}")
    print(f"✓ Missing enrollment: {df['enrollment'].isna().sum()} ({df['enrollment'].isna().mean()*100:.1f}%)")
    print(f"✓ Complete enrollment records: {df['enrollment'].notna().sum()}")
    print()
    
    return df

#==============================================================================
# 2. SAMPLE ATTRITION TABLE
#==============================================================================

def create_sample_attrition_table(df):
    """Create master table showing sample filtering steps"""
    print("=" * 80)
    print("CREATING SAMPLE ATTRITION TABLE")
    print("=" * 80)
    
    attrition = []
    
    # Step 1: Total observations
    attrition.append({
        'Filter_Step': '1. All institution-year observations',
        'Institutions': df['institution_id'].nunique(),
        'Observations': len(df)
    })
    
    # Step 2: Target present
    df_target = df[df['enrollment'].notna()].copy()
    attrition.append({
        'Filter_Step': '2. Target variable (enrollment) present',
        'Institutions': df_target['institution_id'].nunique(),
        'Observations': len(df_target)
    })
    
    # Step 3: Training period
    df_train = df_target[df_target['year'].between(TRAIN_START, TRAIN_END)]
    attrition.append({
        'Filter_Step': '3. Training period (2010-2017)',
        'Institutions': df_train['institution_id'].nunique(),
        'Observations': len(df_train)
    })
    
    # Step 4: Test period with lags
    df_test = df_target[df_target['year'].between(TEST_START, TEST_END)]
    df_test = df_test.sort_values(['institution_id', 'year'])
    df_test['enrollment_lag1'] = df_test.groupby('institution_id')['enrollment'].shift(1)
    df_test_with_lags = df_test[df_test['enrollment_lag1'].notna()]
    attrition.append({
        'Filter_Step': '4. Test period with required lags (2018-2021)',
        'Institutions': df_test_with_lags['institution_id'].nunique(),
        'Observations': len(df_test_with_lags)
    })
    
    attrition_df = pd.DataFrame(attrition)
    attrition_df.to_csv('table_4.0_sample_attrition.csv', index=False)
    print("✓ Table 4.0 saved: table_4.0_sample_attrition.csv")
    print(attrition_df.to_string(index=False))
    print()
    
    return attrition_df

#==============================================================================
# 3. DESCRIPTIVE STATISTICS
#==============================================================================

def compute_descriptive_statistics(df):
    """Compute Table 4.1: Descriptive Statistics"""
    print("=" * 80)
    print("COMPUTING DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
    df_complete = df[df['enrollment'].notna()].copy()
    
    desc_stats = {
        'Statistic': ['Count', 'Mean', 'Std Dev', 'Min', 'Q1', 'Median', 'Q3', 'Max'],
        'Value': [
            len(df_complete),
            df_complete['enrollment'].mean(),
            df_complete['enrollment'].std(),
            df_complete['enrollment'].min(),
            df_complete['enrollment'].quantile(0.25),
            df_complete['enrollment'].median(),
            df_complete['enrollment'].quantile(0.75),
            df_complete['enrollment'].max()
        ]
    }
    
    desc_df = pd.DataFrame(desc_stats)
    desc_df['Value'] = desc_df['Value'].round(0).astype(int)
    
    desc_df.to_csv('table_4.1_descriptive_statistics.csv', index=False)
    print("✓ Table 4.1 saved: table_4.1_descriptive_statistics.csv")
    print(desc_df.to_string(index=False))
    print()
    
    return desc_df

#==============================================================================
# 4. TEMPORAL TRENDS
#==============================================================================

def analyze_temporal_trends(df):
    """Compute Table 4.2: Annual Enrollment Statistics"""
    print("=" * 80)
    print("ANALYZING TEMPORAL TRENDS")
    print("=" * 80)
    
    df_complete = df[df['enrollment'].notna()].copy()
    
    annual_stats = df_complete.groupby('year').agg({
        'institution_id': 'count',
        'enrollment': ['sum', 'mean']
    }).round(1)
    
    annual_stats.columns = ['Institutions', 'Total_Enrollment', 'Average_Enrollment']
    annual_stats = annual_stats.reset_index()
    
    annual_stats.to_csv('table_4.2_annual_statistics.csv', index=False)
    print("✓ Table 4.2 saved: table_4.2_annual_statistics.csv")
    print(annual_stats.to_string(index=False))
    print()
    
    return annual_stats

#==============================================================================
# 5. BASELINE FORECASTING
#==============================================================================

def baseline_forecasting(df):
    """Naive persistence and moving average forecasting"""
    print("=" * 80)
    print("BASELINE FORECASTING")
    print("=" * 80)
    
    df_complete = df[df['enrollment'].notna()].copy()
    df_panel = df_complete.pivot(index='institution_id', columns='year', values='enrollment')
    
    results = []
    
    for test_year in range(TEST_START, TEST_END + 1):
        if test_year - 1 in df_panel.columns and test_year in df_panel.columns:
            forecast = df_panel[test_year - 1]
            actual = df_panel[test_year]
            
            mask = forecast.notna() & actual.notna()
            forecast_clean = forecast[mask]
            actual_clean = actual[mask]
            
            if len(actual_clean) > 0:
                mae = mean_absolute_error(actual_clean, forecast_clean)
                rmse = np.sqrt(mean_squared_error(actual_clean, forecast_clean))
                mape_mean = mean_absolute_percentage_error(actual_clean, forecast_clean) * 100
                mape_median = np.median(np.abs((actual_clean - forecast_clean) / actual_clean) * 100)
                
                results.append({
                    'Year': test_year,
                    'MAE': round(mae, 2),
                    'MAPE_Mean': round(mape_mean, 2),
                    'MAPE_Median': round(mape_median, 2),
                    'RMSE': round(rmse, 2),
                    'N_Institutions': len(actual_clean)
                })
    
    baseline_df = pd.DataFrame(results)
    baseline_df.to_csv('table_4.3_baseline_performance.csv', index=False)
    print("✓ Table 4.3 saved: table_4.3_baseline_performance.csv")
    print(baseline_df.to_string(index=False))
    print()
    
    return baseline_df

#==============================================================================
# MAIN EXECUTION
#==============================================================================

def main():
    """Main execution function"""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + "  ENROLLMENT FORECASTING ANALYSIS - REAL IPEDS DATA".center(78) + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")
    
    # Load data
    df = load_and_prepare_data()
    
    # Run analyses
    attrition_table = create_sample_attrition_table(df)
    desc_stats = compute_descriptive_statistics(df)
    annual_stats = analyze_temporal_trends(df)
    baseline_df = baseline_forecasting(df)
    
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nKey Findings:")
    print(f"✓ Total observations: {len(df)}")
    print(f"✓ Missing rate: {df['enrollment'].isna().mean()*100:.1f}%")
    print(f"✓ Mean enrollment: {df['enrollment'].mean():.0f} students")
    print(f"✓ Median enrollment: {df['enrollment'].median():.0f} students")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
