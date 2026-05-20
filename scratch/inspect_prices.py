import pandas as pd
import numpy as np

def inspect_file(filepath, name):
    print(f"\n=================== Inspecting {name} ({filepath}) ===================")
    try:
        df = pd.read_parquet(filepath)
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print("\nNull count per column:")
        print(df.isnull().sum())
        
        print("\nZero count per column:")
        for col in df.select_dtypes(include=[np.number]).columns:
            zero_count = (df[col] == 0).sum()
            print(f"  {col}: {zero_count} zeros")
            
        print("\nDescriptive statistics:")
        print(df.describe())
        
        # Check ticker 000050 specifically if it exists
        df['ticker_str'] = df['ticker'].astype(str)
        ticker_data = df[df['ticker_str'].str.contains('000050')]
        if not ticker_data.empty:
            print(f"\nData for ticker containing '000050' (found {len(ticker_data)} rows):")
            # Sort by date
            if 'date' in ticker_data.columns:
                ticker_data = ticker_data.sort_values('date')
            print(ticker_data.to_string())
        else:
            print("\nTicker containing '000050' not found in this file.")
            # Print unique tickers as example
            print(f"Sample tickers in dataset: {df['ticker'].unique()[:10]}")
    except Exception as e:
        print(f"Error inspecting {name}: {e}")

if __name__ == '__main__':
    inspect_file('/Users/mireflare/Documents/Codes/quant-ml/data/raw/prices/prices_kr_raw.parquet', 'prices_kr_raw')
    inspect_file('/Users/mireflare/Documents/Codes/quant-ml/data/processed/prices_quarterly.parquet', 'prices_quarterly')
