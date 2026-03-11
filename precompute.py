"""
precompute.py — Run once to generate analysis_data.json
Usage: python3 precompute.py
"""
import json, sys, time
sys.path.insert(0, '.')
from analysis.engine import run_full_pipeline

CSV = 'data/rossmann_sales.csv'
OUT = 'static/analysis_data.json'

all_data = {}
for store_id in [1, 2, 3]:
    print(f"  Computing Store {store_id}...", end=' ', flush=True)
    t = time.time()
    result = run_full_pipeline(CSV, store_id=store_id)
    all_data[str(store_id)] = result
    print(f"{time.time()-t:.0f}s")

with open(OUT, 'w') as f:
    json.dump(all_data, f, separators=(',', ':'))

size_kb = len(json.dumps(all_data)) // 1024
print(f"\nSaved to {OUT}  ({size_kb} KB)")
print("Done — server can now serve instantly from cache.")
