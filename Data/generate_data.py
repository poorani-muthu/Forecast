"""
Generate a realistic multi-store retail sales dataset modelled on Rossmann.
~5 years of daily data, 3 stores, with:
  - Upward trend
  - Weekly seasonality (Mon-Sun pattern)
  - Annual seasonality (Christmas peak, summer dip, Jan trough)
  - Promotions (lift ~15%)
  - Public holidays (drop ~60%)
  - School holidays (slight lift)
  - Structural break / COVID-style dip in year 4
  - Additive noise
"""
import pandas as pd
import numpy as np
from datetime import date, timedelta

np.random.seed(2024)

START = date(2018, 1, 1)
END   = date(2023, 12, 31)
STORES = [1, 2, 3]

# Store baselines
STORE_BASE = {1: 6500, 2: 5200, 3: 7800}
STORE_TREND = {1: 0.00012, 2: 0.00008, 3: 0.00015}  # daily fractional trend

# Weekly pattern (Mon=0 .. Sun=6)
WEEKLY = {0:1.00, 1:1.05, 2:1.02, 3:1.08, 4:1.15, 5:1.35, 6:0.00}  # Sun closed

# Monthly seasonality index
MONTHLY = {1:0.82, 2:0.88, 3:0.95, 4:0.97, 5:1.00, 6:0.96,
           7:0.93, 8:0.97, 9:1.02, 10:1.05, 11:1.12, 12:1.28}

# German public holidays (approx)
def get_holidays(year):
    h = set()
    h.add(date(year, 1, 1))   # New Year
    h.add(date(year, 5, 1))   # Labour Day
    h.add(date(year, 10, 3))  # German Unity
    h.add(date(year, 12, 25)) # Christmas
    h.add(date(year, 12, 26)) # Boxing Day
    # Easter (approx)
    a = year % 19; b = year // 100; c = year % 100
    d2 = (19*a + b - b//4 - (b - (b+8)//25 + 1)//3 + 15) % 30
    e = (32 + 2*(b%4) + 2*(c//4) - d2 - (c%4)) % 7
    f = d2 + e - 7*((a + 11*d2 + 22*e)//451) + 114
    month = f // 31; day = f % 31 + 1
    from datetime import date as dt
    easter = dt(year, month, day)
    h.add(easter)
    h.add(easter + timedelta(days=1))  # Easter Monday
    h.add(easter - timedelta(days=2))  # Good Friday
    return h

all_holidays = set()
for y in range(2018, 2024):
    all_holidays |= get_holidays(y)

rows = []
days = (END - START).days + 1

for store_id in STORES:
    base = STORE_BASE[store_id]
    trend_rate = STORE_TREND[store_id]

    for i in range(days):
        current = START + timedelta(days=i)
        dow = current.weekday()  # 0=Mon

        # Sunday closed
        if dow == 6:
            continue

        t = i
        sales = base * (1 + trend_rate * t)

        # Weekly
        sales *= WEEKLY[dow]

        # Monthly
        sales *= MONTHLY[current.month]

        # Holiday
        state_holiday = 0
        school_holiday = 0
        if current in all_holidays:
            sales *= 0.40
            state_holiday = 1
        else:
            # School holiday: July-Aug, Dec last 2 weeks
            if current.month in [7, 8]:
                sales *= 1.04
                school_holiday = 1
            elif current.month == 12 and current.day >= 18:
                sales *= 1.10
                school_holiday = 1

        # Pre-holiday spike (day before holiday)
        next_day = current + timedelta(days=1)
        if next_day in all_holidays:
            sales *= 1.18

        # Promotion (random ~30% of days, bigger on weekends)
        promo = 0
        if np.random.random() < 0.30:
            promo = 1
            sales *= 1.15 + np.random.uniform(-0.03, 0.05)

        # COVID-style disruption: Mar-Jun 2020
        if date(2020, 3, 15) <= current <= date(2020, 6, 14):
            sales *= 0.45

        # Post-COVID recovery ramp Jun-Dec 2020
        if date(2020, 6, 15) <= current <= date(2020, 12, 31):
            recovery = (current - date(2020, 6, 15)).days / 200
            sales *= 0.50 + min(recovery, 0.50)

        # Store-specific noise
        noise_scale = 0.08 if store_id == 1 else 0.07 if store_id == 2 else 0.09
        sales *= np.random.lognormal(0, noise_scale)

        rows.append({
            'Date':          current.strftime('%Y-%m-%d'),
            'Store':         store_id,
            'Sales':         max(0, round(sales, 2)),
            'DayOfWeek':     dow + 1,   # 1=Mon
            'Promo':         promo,
            'StateHoliday':  state_holiday,
            'SchoolHoliday': school_holiday,
            'Month':         current.month,
            'Year':          current.year,
            'WeekOfYear':    current.isocalendar()[1],
        })

df = pd.DataFrame(rows)
out = '/home/claude/timeseries_forecast/data/rossmann_sales.csv'
df.to_csv(out, index=False)
print(f"Dataset: {len(df):,} rows  |  {df['Store'].nunique()} stores  |  {df['Date'].min()} → {df['Date'].max()}")
print(df.groupby('Store')['Sales'].describe().round(0))
