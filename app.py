"""
app.py — Time Series Forecasting Dashboard
Run:  python3 app.py
Open: http://localhost:5000
"""
from flask import Flask, jsonify, render_template, request
import json, os

app = Flask(__name__)

# ── Load pre-computed cache at startup ──────────────────────────────────────
CACHE_PATH = os.path.join(os.path.dirname(__file__), 'static', 'analysis_data.json')

def load_cache():
    if not os.path.exists(CACHE_PATH):
        return None
    with open(CACHE_PATH) as f:
        return json.load(f)

CACHE = load_cache()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def api_data():
    if CACHE is None:
        return jsonify({'error': 'Run precompute.py first'}), 500
    store = request.args.get('store', '1')
    if store not in CACHE:
        store = '1'
    return jsonify(CACHE[store])

@app.route('/api/stores')
def api_stores():
    if CACHE is None:
        return jsonify([])
    return jsonify([
        {'id': sid, 'label': f'Store {sid}',
         'mean': round(CACHE[sid]['audit']['mean_sales'], 0),
         'total': CACHE[sid]['audit']['total_sales']}
        for sid in CACHE
    ])

if __name__ == '__main__':
    if CACHE is None:
        print("ERROR: static/analysis_data.json not found.")
        print("Run:  python3 precompute.py  first.")
    else:
        print("=" * 55)
        print("  Time Series Forecasting Dashboard — Poorani M")
        print("=" * 55)
        print("  Server  :  http://localhost:5000")
        print("  Stores  :  3 stores, 5 years of daily data")
        print("  Models  :  Holt-Winters · Gradient Boosting · Seasonal Naive")
        print("=" * 55)
    app.run(debug=False, host='0.0.0.0', port=5000)
