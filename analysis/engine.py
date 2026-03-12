"""
analysis/engine.py
Full time-series analysis pipeline — no statsmodels / prophet required.

Models implemented from scratch:
  1. Seasonal Decomposition (additive, period=7)
  2. Holt-Winters Exponential Smoothing (triple, additive)
  3. SARIMA-equivalent via OLS with Fourier + lag features (sklearn)
  4. ML Gradient Boosting (sklearn GradientBoostingRegressor)

Evaluation:
  - Walk-Forward (expanding window) validation — NOT random split
  - RMSE, MAE, MAPE per model
  - Forecast 30 days ahead with 95% prediction interval

Key explainers embedded:
  - Why random train/test split is WRONG for time series
  - Seasonality, trend, residual interpretation
  - Model drift detection (rolling MAE)
"""
import pandas as pd
import numpy as np
import warnings, json
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════════════════════════
#  UTILS
# ══════════════════════════════════════════════════════════════════════════════

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2)))

def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def smape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom != 0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100)


# ══════════════════════════════════════════════════════════════════════════════
#  1. DATA LOADING & PREP
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prep(csv_path: str, store_id: int = 1) -> dict:
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Store'] == store_id].sort_values('Date').reset_index(drop=True)

    # Aggregate to daily (already daily, but ensure no gaps)
    df = df.set_index('Date').asfreq('B', fill_value=None)  # business days
    df = df.reset_index()
    df = df.sort_values('Date').reset_index(drop=True)

    # Forward-fill any gaps
    df['Sales'] = df['Sales'].ffill().fillna(0)

    audit = {
        'store_id':    store_id,
        'n_rows':      len(df),
        'date_start':  str(df['Date'].min().date()),
        'date_end':    str(df['Date'].max().date()),
        'mean_sales':  round(float(df['Sales'].mean()), 2),
        'max_sales':   round(float(df['Sales'].max()), 2),
        'min_sales':   round(float(df['Sales'].min()), 2),
        'total_sales': round(float(df['Sales'].sum()), 2),
        'n_stores':    int(pd.read_csv(csv_path)['Store'].nunique()),
    }
    return {'df': df, 'audit': audit}


# ══════════════════════════════════════════════════════════════════════════════
#  2. SEASONAL DECOMPOSITION (additive, moving-average)
# ══════════════════════════════════════════════════════════════════════════════

def seasonal_decompose(series: np.ndarray, period: int = 7) -> dict:
    """Additive decomposition: Y = Trend + Seasonal + Residual"""
    n = len(series)

    # Trend via centred moving average
    half = period // 2
    trend = np.full(n, np.nan)
    for i in range(half, n - half):
        trend[i] = np.mean(series[i - half: i + half + 1])

    # Detrend
    detrended = series - trend

    # Seasonal indices (average per period position)
    seasonal_avg = np.zeros(period)
    counts = np.zeros(period)
    for i in range(n):
        if not np.isnan(detrended[i]):
            seasonal_avg[i % period] += detrended[i]
            counts[i % period] += 1
    counts[counts == 0] = 1
    seasonal_avg /= counts
    seasonal_avg -= seasonal_avg.mean()  # centre

    seasonal = np.array([seasonal_avg[i % period] for i in range(n)])

    # Residual
    residual = series - trend - seasonal
    trend_clean = np.where(np.isnan(trend), np.nanmean(trend), trend)

    return {
        'original':  series.tolist(),
        'trend':     trend_clean.tolist(),
        'seasonal':  seasonal.tolist(),
        'residual':  residual.tolist(),
        'seasonal_indices': seasonal_avg.tolist(),
        'period':    period,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  3. HOLT-WINTERS TRIPLE EXPONENTIAL SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════

def holt_winters_fit(series: np.ndarray, period: int = 7,
                     alpha: float = 0.3, beta: float = 0.1, gamma: float = 0.2):
    """Additive Holt-Winters. Returns fitted values."""
    n = len(series)
    # Initialise
    L = np.mean(series[:period])
    b = (np.mean(series[period:2*period]) - np.mean(series[:period])) / period
    S = [(series[i] - L) for i in range(period)]
    S_mean = sum(S) / period
    S = [s - S_mean for s in S]

    fitted = []
    for t in range(n):
        if t == 0:
            fitted.append(L + b + S[0])
            continue
        prev_L, prev_b = L, b
        prev_S = S[t % period]
        L = alpha * (series[t] - prev_S) + (1 - alpha) * (prev_L + prev_b)
        b = beta  * (L - prev_L)         + (1 - beta)  * prev_b
        S[t % period] = gamma * (series[t] - L) + (1 - gamma) * prev_S
        fitted.append(L + b + S[t % period])

    return np.array(fitted), L, b, S


def holt_winters_forecast(L, b, S, steps: int, period: int = 7):
    n = len(S)
    forecast = []
    for h in range(1, steps + 1):
        f = L + h * b + S[(n - period + h) % period]
        forecast.append(max(0, f))
    return np.array(forecast)


# ══════════════════════════════════════════════════════════════════════════════
#  4. FOURIER + LAG RIDGE REGRESSION (SARIMA proxy)
# ══════════════════════════════════════════════════════════════════════════════

def build_fourier_features(t: np.ndarray, period: float, n_harmonics: int = 3):
    feats = []
    for k in range(1, n_harmonics + 1):
        feats.append(np.sin(2 * np.pi * k * t / period))
        feats.append(np.cos(2 * np.pi * k * t / period))
    return np.column_stack(feats)


def build_sarima_features(df: pd.DataFrame, lags=(1, 2, 3, 7, 14)):
    series = df['Sales'].values
    n = len(series)
    t = np.arange(n)

    # Fourier terms for weekly (7) and annual (365) seasonality
    F7   = build_fourier_features(t, 7,   n_harmonics=3)
    F365 = build_fourier_features(t, 365, n_harmonics=2)

    # Time trend + squared
    t_norm = t / n
    trend_feats = np.column_stack([t_norm, t_norm**2])

    # Lag features
    lag_feats = np.column_stack([
        np.roll(series, lag) for lag in lags
    ])
    # Zero out invalid lags
    max_lag = max(lags)
    lag_feats[:max_lag, :] = np.nan

    # External regressors
    ext_feats = df[['Promo', 'StateHoliday', 'SchoolHoliday']].fillna(0).values

    X = np.hstack([F7, F365, trend_feats, lag_feats, ext_feats])
    y = series

    # Drop rows with NaN (from lags)
    valid = ~np.any(np.isnan(X), axis=1)
    return X, y, valid, max_lag


def fit_sarima_proxy(df, train_idx):
    X, y, valid, max_lag = build_sarima_features(df)
    train_mask = valid & (np.arange(len(df)) < train_idx[-1] + 1) & (np.arange(len(df)) >= max_lag)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_mask])
    y_train = y[train_mask]
    model = Ridge(alpha=10.0)
    model.fit(X_train, y_train)
    return model, scaler, X, y, valid, max_lag


# ══════════════════════════════════════════════════════════════════════════════
#  5. ML GRADIENT BOOSTING
# ══════════════════════════════════════════════════════════════════════════════

def build_ml_features(df: pd.DataFrame):
    series = df['Sales'].values
    n = len(series)
    dates = pd.to_datetime(df['Date'])

    dow   = dates.dt.dayofweek.values
    month = dates.dt.month.values
    week  = dates.dt.isocalendar().week.values.astype(int)
    year  = dates.dt.year.values
    t     = np.arange(n)

    # Cyclical encoding
    dow_sin = np.sin(2*np.pi*dow/7)
    dow_cos = np.cos(2*np.pi*dow/7)
    mon_sin = np.sin(2*np.pi*month/12)
    mon_cos = np.cos(2*np.pi*month/12)
    wk_sin  = np.sin(2*np.pi*week/52)
    wk_cos  = np.cos(2*np.pi*week/52)

    lags = [1, 2, 3, 7, 14, 21, 28]
    lag_feats = np.column_stack([np.roll(series, l) for l in lags])

    # Rolling stats (7-day, 28-day)
    roll7  = pd.Series(series).rolling(7,  min_periods=1).mean().values
    roll28 = pd.Series(series).rolling(28, min_periods=1).mean().values
    roll7_std = pd.Series(series).rolling(7, min_periods=1).std().fillna(0).values

    ext = df[['Promo','StateHoliday','SchoolHoliday']].fillna(0).values
    t_norm = (t / n).reshape(-1,1)
    year_norm = ((year - year.min()) / max(year.max() - year.min(), 1)).reshape(-1,1)

    X = np.hstack([
        dow_sin.reshape(-1,1), dow_cos.reshape(-1,1),
        mon_sin.reshape(-1,1), mon_cos.reshape(-1,1),
        wk_sin.reshape(-1,1),  wk_cos.reshape(-1,1),
        t_norm, year_norm,
        lag_feats, roll7.reshape(-1,1), roll28.reshape(-1,1), roll7_std.reshape(-1,1),
        ext,
    ])

    max_lag = max(lags)
    valid = np.ones(n, dtype=bool)
    valid[:max_lag] = False

    return X, series, valid, max_lag


# ══════════════════════════════════════════════════════════════════════════════
#  6. WALK-FORWARD VALIDATION (expanding window)
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_validation(series: np.ndarray, model_fn, n_test: int = 90,
                             min_train: int = 200, step: int = 7):
    """
    Expanding window: train on [0..t], predict [t..t+step], advance.
    Returns arrays of true and predicted values for the test window.
    """
    n = len(series)
    train_end = n - n_test
    all_true, all_pred = [], []

    t = min_train
    while t < train_end:
        window_end = min(t + step, train_end)
        y_train = series[:t]
        y_true_window = series[t:window_end]
        horizon = len(y_true_window)

        try:
            y_pred_window = model_fn(y_train, horizon)
            all_true.extend(y_true_window.tolist())
            all_pred.extend(y_pred_window.tolist())
        except Exception:
            pass

        t = window_end

    return np.array(all_true), np.array(all_pred)


def hw_model_fn(y_train, horizon):
    _, L, b, S = holt_winters_fit(y_train)
    return holt_winters_forecast(L, b, S, horizon)


def naive_seasonal_fn(y_train, horizon, period=7):
    """Seasonal naive: forecast = same day last week."""
    preds = []
    for h in range(1, horizon + 1):
        idx = len(y_train) - period + ((h - 1) % period)
        preds.append(max(0, y_train[idx]))
    return np.array(preds)


# ══════════════════════════════════════════════════════════════════════════════
#  7. DRIFT DETECTION (rolling MAE)
# ══════════════════════════════════════════════════════════════════════════════

def compute_drift(y_true, y_pred, window=14):
    """Rolling MAE to detect model drift over time."""
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    rolling = pd.Series(errors).rolling(window, min_periods=1).mean()
    return rolling.tolist()


# ══════════════════════════════════════════════════════════════════════════════
#  8. FORECASTING (30-day ahead)
# ══════════════════════════════════════════════════════════════════════════════

def generate_forecast(df, series, n_forecast=30):
    """
    Fit Holt-Winters on full series, forecast 30 days ahead.
    Prediction interval via residual std (parametric).
    """
    _, L, b, S = holt_winters_fit(series, period=7)
    point = holt_winters_forecast(L, b, S, n_forecast, period=7)

    # Estimate residual std from last 90 days
    fitted, *_ = holt_winters_fit(series[-90:])
    resid_std  = np.std(series[-90:] - fitted)

    z = 1.96
    lower = np.maximum(0, point - z * resid_std)
    upper = point + z * resid_std

    last_date = pd.to_datetime(df['Date'].iloc[-1])
    future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_forecast)

    return {
        'dates':  [str(d.date()) for d in future_dates],
        'point':  [round(float(v), 2) for v in point],
        'lower':  [round(float(v), 2) for v in lower],
        'upper':  [round(float(v), 2) for v in upper],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  9. MASTER RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(csv_path: str, store_id: int = 1) -> dict:
    # ── Load ──
    prep = load_and_prep(csv_path, store_id)
    df, audit = prep['df'], prep['audit']
    series = df['Sales'].values.astype(float)
    dates  = df['Date'].dt.strftime('%Y-%m-%d').tolist()
    n      = len(series)

    # ── Decomposition ──
    decomp = seasonal_decompose(series, period=7)

    # ── Monthly aggregation ──
    df['YM'] = df['Date'].dt.to_period('M').astype(str)
    monthly = df.groupby('YM')['Sales'].sum().reset_index()
    monthly.columns = ['Month', 'Sales']

    # ── Weekly aggregation ──
    df['Week'] = df['Date'].dt.to_period('W').astype(str)
    weekly = df.groupby('Week')['Sales'].sum().reset_index()
    weekly.columns = ['Week', 'Sales']

    # ── Day-of-week profile ──
    dow_labels = ['Mon','Tue','Wed','Thu','Fri','Sat']
    df['DOW_label'] = df['Date'].dt.dayofweek.map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat'})
    dow_avg = df.groupby('DOW_label')['Sales'].mean().reindex(dow_labels).fillna(0)

    # ── Monthly profile ──
    mon_labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    df['MON_label'] = df['Date'].dt.month.map(dict(enumerate(mon_labels, 1)))
    mon_avg = df.groupby('MON_label')['Sales'].mean().reindex(mon_labels).fillna(0)

    # ── Walk-forward validation ──
    n_test   = 90
    test_series = series

    hw_true, hw_pred = walk_forward_validation(test_series, hw_model_fn, n_test=n_test, min_train=180)
    sn_true, sn_pred = walk_forward_validation(test_series, naive_seasonal_fn, n_test=n_test, min_train=180)

    # ML validation (manual expanding window for GBR)
    def gbr_expanding(full_series, full_df, n_test=90, min_train=180, step=7):
        X_all, y_all, valid, max_lag = build_ml_features(full_df)
        n = len(full_series)
        train_end = n - n_test
        all_true, all_pred = [], []
        t = max(min_train, max_lag + 1)
        while t < train_end:
            window_end = min(t + step, train_end)
            train_mask = valid & (np.arange(n) < t)
            test_mask  = np.arange(n)
            test_idx   = np.arange(t, window_end)
            valid_test = [i for i in test_idx if valid[i]]
            if sum(train_mask) < 50 or len(valid_test) == 0:
                t = window_end; continue
            gbr = GradientBoostingRegressor(n_estimators=80, max_depth=4,
                                            learning_rate=0.08, random_state=42)
            gbr.fit(X_all[train_mask], y_all[train_mask])
            preds = gbr.predict(X_all[valid_test])
            trues = y_all[valid_test]
            all_true.extend(trues.tolist())
            all_pred.extend(np.maximum(0, preds).tolist())
            t = window_end
        return np.array(all_true), np.array(all_pred)

    ml_true, ml_pred = gbr_expanding(series, df, n_test=n_test, min_train=180)

    # ── Metrics ──
    def metrics(yt, yp):
        return {
            'rmse':  round(rmse(yt, yp), 2),
            'mae':   round(mae(yt, yp), 2),
            'mape':  round(mape(yt, yp), 2),
            'smape': round(smape(yt, yp), 2),
        }

    model_metrics = {
        'Holt-Winters':       metrics(hw_true, hw_pred),
        'Seasonal Naive':     metrics(sn_true, sn_pred),
        'Gradient Boosting':  metrics(ml_true, ml_pred) if len(ml_true) else {'rmse':0,'mae':0,'mape':0,'smape':0},
    }

    # Best model
    best_model = min(model_metrics, key=lambda m: model_metrics[m]['rmse'])

    # ── Drift simulation ──
    drift_mae = compute_drift(hw_true, hw_pred, window=14)
    drift_dates = dates[-(len(drift_mae)):]

    # ── Walk-forward comparison data (last 90 days for chart) ──
    hw_chart_dates  = dates[-len(hw_true):]
    ml_chart_dates  = dates[-len(ml_true):]

    # ── Forecast ──
    forecast = generate_forecast(df, series, n_forecast=30)

    # ── Leakage explainer data ──
    split_demo = _leakage_demo(series)

    # ── All stores summary ──
    all_df = pd.read_csv(csv_path)
    all_df['Date'] = pd.to_datetime(all_df['Date'])
    store_summary = all_df.groupby('Store').agg(
        TotalSales=('Sales','sum'),
        AvgDailySales=('Sales','mean'),
        MaxSales=('Sales','max'),
        Days=('Sales','count')
    ).reset_index()

    return {
        'audit':          audit,
        'dates':          dates,
        'sales':          [round(float(v),2) for v in series],
        'monthly':        monthly.to_dict(orient='records'),
        'weekly':         weekly.tail(104).to_dict(orient='records'),
        'dow_avg':        {'labels': dow_labels, 'values': [round(float(v),2) for v in dow_avg]},
        'mon_avg':        {'labels': mon_labels, 'values': [round(float(v),2) for v in mon_avg]},
        'decomp':         {
            'dates':     dates,
            'original':  decomp['original'],
            'trend':     decomp['trend'],
            'seasonal':  decomp['seasonal'],
            'residual':  decomp['residual'],
            'period':    decomp['period'],
            'seasonal_indices': decomp['seasonal_indices'],
        },
        'hw_pred':        [round(float(v),2) for v in hw_pred],
        'hw_true':        [round(float(v),2) for v in hw_true],
        'hw_dates':       hw_chart_dates,
        'ml_pred':        [round(float(v),2) for v in ml_pred],
        'ml_true':        [round(float(v),2) for v in ml_true],
        'ml_dates':       ml_chart_dates,
        'sn_pred':        [round(float(v),2) for v in sn_pred],
        'model_metrics':  model_metrics,
        'best_model':     best_model,
        'drift_mae':      [round(float(v),2) for v in drift_mae],
        'drift_dates':    drift_dates,
        'forecast':       forecast,
        'split_demo':     split_demo,
        'store_summary':  store_summary.to_dict(orient='records'),
        'n_test':         n_test,
    }


def _leakage_demo(series):
    """
    Demonstrate why random split causes data leakage in time series.
    Returns naive random-split metrics vs walk-forward metrics.
    """
    n = len(series)
    np.random.seed(42)

    # Random split "cheating" model — shuffle indices
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, test_idx = np.sort(idx[:split]), np.sort(idx[split:])

    # Predict test as mean of training + seasonal (cheating — it has future info)
    cheat_preds = np.full(len(test_idx), np.mean(series[train_idx]))
    cheat_true  = series[test_idx]
    cheat_rmse  = round(rmse(cheat_true, cheat_preds), 2)
    cheat_mape  = round(mape(cheat_true, cheat_preds), 2)

    # Honest walk-forward
    hw_t, hw_p = walk_forward_validation(series, hw_model_fn, n_test=90, min_train=180)
    honest_rmse = round(rmse(hw_t, hw_p), 2)
    honest_mape = round(mape(hw_t, hw_p), 2)

    return {
        'random_split_rmse': cheat_rmse,
        'random_split_mape': cheat_mape,
        'walkforward_rmse':  honest_rmse,
        'walkforward_mape':  honest_mape,
        'explanation': (
            "Random split shuffles dates — the model trains on FUTURE data to predict the PAST. "
            "This leaks future information and gives falsely optimistic metrics. "
            "Walk-forward validation always trains only on past data, simulating real deployment."
        ),
    }
