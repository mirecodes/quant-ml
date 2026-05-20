import pandas as pd
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.validation import check_intermediate_gaps

if __name__ == '__main__':
    filepath = '/Users/mireflare/Documents/Codes/quant-ml/data/processed/prices_quarterly.parquet'
    try:
        df = pd.read_parquet(filepath)
        check_intermediate_gaps(df, "prices_quarterly.parquet")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
