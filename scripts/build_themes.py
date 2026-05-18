#!/usr/bin/env python3
import os
import sys
import glob
from pathlib import Path
import yaml
from yaml.resolver import Resolver

# Disable boolean parsing for ON, NO, etc. which are valid stock tickers
for char in list("yYnNoOtTfF"):
    if char in Resolver.yaml_implicit_resolvers:
        Resolver.yaml_implicit_resolvers[char] = [
            (tag, regexp) for tag, regexp in Resolver.yaml_implicit_resolvers[char]
            if tag != 'tag:yaml.org,2002:bool'
        ]

def main():
    print("=== Processing and Merging Global Themes ===")
    
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / 'themes' / 'raw'
    processed_dir = project_root / 'themes' / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    global_themes_path = raw_dir / 'global_themes.yaml'
    if not global_themes_path.exists():
        print(f"Error: global_themes.yaml not found at {global_themes_path}")
        sys.exit(1)
        
    with open(global_themes_path, 'r', encoding='utf-8') as f:
        global_themes = yaml.safe_load(f)
        
    # Build valid theme set and their metadata
    valid_themes = {}
    for key, val in global_themes.items():
        if isinstance(val, dict) and 'tier' in val:
            valid_themes[key] = val
            
    print(f"Loaded {len(valid_themes)} defined themes from global_themes.yaml.")
    
    # Track stats
    stats = {
        'kr_tickers': set(),
        'us_tickers': set(),
        'undefined_theme_errors': 0,
        'unique_themes_used': set()
    }
    
    merged_mappings = {}
    
    # Helper to load and merge mapping files
    def load_mappings(subdir, country):
        search_path = os.path.join(raw_dir / subdir, "*.yaml")
        part_files = sorted(glob.glob(search_path))
        
        if not part_files:
            print(f"Warning: No part files found in {raw_dir / subdir}")
            return
            
        for path in part_files:
            print(f"Loading {os.path.basename(path)}...")
            with open(path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                if not content:
                    continue
                    
                for ticker_raw, data in content.items():
                    # Clean _DUP suffix
                    ticker = ticker_raw.split('_DUP')[0]
                    
                    if ticker not in merged_mappings:
                        merged_mappings[ticker] = {
                            'name': data.get('name', ''),
                            'country': country,
                            'themes': []
                        }
                    
                    # Merge themes (avoid duplicates)
                    for theme_id in data.get('themes', []):
                        if theme_id not in valid_themes:
                            print(f"Error: Undefined theme ID '{theme_id}' found in {os.path.basename(path)} for ticker '{ticker}'")
                            stats['undefined_theme_errors'] += 1
                        else:
                            if theme_id not in merged_mappings[ticker]['themes']:
                                merged_mappings[ticker]['themes'].append(theme_id)
                                stats['unique_themes_used'].add(theme_id)
                                
                    if country == 'KR':
                        stats['kr_tickers'].add(ticker)
                    else:
                        stats['us_tickers'].add(ticker)

    # Process KOSPI and S&P 500
    load_mappings('kospi', 'KR')
    load_mappings('sp500', 'US')
    
    # Save output merged mapping
    output_data = {
        'global_themes': valid_themes,
        'mappings': merged_mappings
    }
    
    output_path = processed_dir / 'merged_themes.yaml'
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(output_data, f, allow_unicode=True, sort_keys=False)
        
    print("\n=======================================================")
    print("               Theme Merge Process Complete            ")
    print("=======================================================")
    print(f"KOSPI 고유 종목: {len(stats['kr_tickers'])}개")
    print(f"S&P 500 고유 종목: {len(stats['us_tickers'])}개")
    print(f"전체 커버 종목: {len(merged_mappings)}개")
    print(f"사용된 Tier-3 테마: {len([t for t in stats['unique_themes_used'] if valid_themes[t]['tier'] == 3])}개")
    print(f"정의된 전체 테마 (T1+T2+T3): {len(valid_themes)}개")
    print(f"미정의 테마 ID (오류): {stats['undefined_theme_errors']}개 {'✅' if stats['undefined_theme_errors'] == 0 else '❌'}")
    print(f"Processed file saved to: {output_path}")
    print("=======================================================")

if __name__ == '__main__':
    main()
