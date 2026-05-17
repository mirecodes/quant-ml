# scripts/03_build_labels.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.labels.attractiveness import compute_attractiveness
from src.labels.risk import compute_risk
from src.utils.io import save_parquet, load_parquet, report_memory

def main():
    prices = load_parquet('data/processed/prices_quarterly.parquet')
    report_memory(prices, "prices")
    
    a = compute_attractiveness(prices, max_horizon_years=5)
    r = compute_risk(prices, max_horizon_years=5)
    
    if a.empty or r.empty:
        print("Warning: One of attractiveness or risk label sets is empty.")
        
    labels = pd.DataFrame()
    if not a.empty and not r.empty:
        labels = a.merge(r, on=['ticker', 'date'], how='outer')
    elif not a.empty:
        labels = a
        labels['R'] = 0.0
        labels['R_quarters_used'] = 0
    elif not r.empty:
        labels = r
        labels['A'] = 0.0
        labels['A_quarters_used'] = 0
        
    if not labels.empty:
        save_parquet(labels, 'data/processed/labels.parquet')
        report_memory(labels, "labels")
        print(f"Generated {len(labels)} label rows")
    else:
        print("Error: No labels generated.")

if __name__ == '__main__':
    main()
