# ============================================================
# SBERT-02: DAILY AND MONTHLY COORDINATION INDICES
# KEYWORD VS SBERT COMPARISON
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. SETTINGS
# Put this script in the same folder as the input CSV, or specify its full path.
SPEECH_FILE = Path('speeches8-SBERT-TwoStage.csv')
PREFIX = 'Coordination-Index-SBERT'
NBER_RECESSIONS = [
    ('2001-03-01', '2001-11-30'),
    ('2007-12-01', '2009-06-30'),
    ('2020-02-01', '2020-04-30'),
]

# 2. IMPORT AND VALIDATE
speeches = pd.read_csv(SPEECH_FILE)
required = ['ymd', 'SBERT_Intensity', 'SBERT_90', 'SBERT_90_Binary', 'log_freqtrigger']
missing = set(required) - set(speeches.columns)
if missing:
    raise ValueError(f'Missing columns: {sorted(missing)}')
speeches['ymd'] = pd.to_datetime(speeches['ymd'], errors='raise')
for col in required[1:]:
    speeches[col] = pd.to_numeric(speeches[col], errors='coerce')
print('Speeches:', len(speeches))

# 3. DAILY AGGREGATION: max across speeches on a given day
# The original keyword values, including threshold-generated zeros, are unchanged.
daily = (
    speeches.groupby('ymd', as_index=False)
    .agg(Keyword_Index=('log_freqtrigger', 'max'),
         SBERT_Intensity=('SBERT_Intensity', 'max'),
         SBERT_Index=('SBERT_90', 'max'),
         SBERT_Binary=('SBERT_90_Binary', 'max'))
    .sort_values('ymd').reset_index(drop=True)
)

# 4. LOG SBERT: log of positive values; zeros stay zero; missing stay missing
# This is a log transformation, not log1p.
daily['SBERT_Log'] = np.nan
positive = daily['SBERT_Index'] > 0
zero = daily['SBERT_Index'] == 0
daily.loc[positive, 'SBERT_Log'] = np.log(daily.loc[positive, 'SBERT_Index'])
daily.loc[zero, 'SBERT_Log'] = 0.0

# 5. COMPLETE DAILY CALENDAR: zero only when no speech exists
calendar = pd.DataFrame({'ymd': pd.date_range(daily['ymd'].min(), daily['ymd'].max(), freq='D')})
daily_full = calendar.merge(daily, on='ymd', how='left', validate='one_to_one', indicator=True)
no_speech = daily_full['_merge'].eq('left_only')
columns = ['Keyword_Index', 'SBERT_Intensity', 'SBERT_Index', 'SBERT_Binary', 'SBERT_Log']
daily_full.loc[no_speech, columns] = 0.0
daily_full = daily_full.drop(columns='_merge')
daily_full['ym'] = daily_full['ymd'].dt.to_period('M').astype(str)
print('\nDAILY OBSERVATIONS')
print('Calendar days:', len(daily_full))
print('Speech days:', len(daily))
print('Days without speeches:', int(no_speech.sum()))
print('Positive keyword days:', int((daily_full['Keyword_Index'] > 0).sum()))
print('Positive SBERT days:', int((daily_full['SBERT_Binary'] > 0).sum()))

# 6. MONTHLY AGGREGATION: sums of daily indices
sum_nonmissing = lambda x: x.sum(min_count=1)
monthly = (
    daily_full.groupby('ym', as_index=False)
    .agg(monthlylog_freqtrigger=('Keyword_Index', sum_nonmissing),
         SBERT_Index=('SBERT_Index', sum_nonmissing),
         SBERT_Log=('SBERT_Log', sum_nonmissing),
         SBERT_Binary=('SBERT_Binary', sum_nonmissing))
    .sort_values('ym').reset_index(drop=True)
)
monthly['date'] = pd.to_datetime(monthly['ym'] + '-01')

# 7. EXPORT DATA
daily_full.to_csv(f'{PREFIX}-Daily.csv', index=False)
monthly.to_csv(f'{PREFIX}-Monthly.csv', index=False)

# 8. CORRELATIONS: speech days for daily, all months for monthly
def calculate_correlations(data, keyword, variables):
    rows = []
    for col in variables:
        paired = data[[keyword, col]].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            'measure': col,
            'observations': len(paired),
            'keyword_variance': paired[keyword].var(),
            'sbert_variance': paired[col].var(),
            'correlation': paired[keyword].corr(paired[col]),
        })
    return pd.DataFrame(rows)

variables = ['SBERT_Index', 'SBERT_Log', 'SBERT_Binary']
daily_stats = calculate_correlations(daily_full.loc[~no_speech], 'Keyword_Index', variables)
monthly_stats = calculate_correlations(monthly, 'monthlylog_freqtrigger', variables)
print('\nDAILY KEYWORD-SBERT CORRELATIONS')
print(daily_stats.to_string(index=False))
print('\nMONTHLY KEYWORD-SBERT CORRELATIONS')
print(monthly_stats.to_string(index=False))
daily_stats.to_csv(f'{PREFIX}-Daily-Correlations.csv', index=False)
monthly_stats.to_csv(f'{PREFIX}-Monthly-Correlations.csv', index=False)

# 9. FIGURE FORMATTING: daily YYYY-MM-DD / monthly YYYY-MM
def format_axis(ax, dates, frequency):
    for start, end in NBER_RECESSIONS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), color='gray',
                   alpha=0.22, linewidth=0, zorder=0)
    ax.axhline(0, color='black', linewidth=0.7)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    if frequency == 'daily':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.set_xlabel('Day', fontsize=12)
    elif frequency == 'monthly':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.set_xlabel('Month', fontsize=12)
    else:
        raise ValueError('frequency must be daily or monthly')
    ax.set_xlim(dates.min(), dates.max())
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelrotation=45)

# 10. DAILY FIGURE
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
keyword_plot = daily_full.loc[np.isfinite(daily_full['Keyword_Index'])]
axes[0].vlines(keyword_plot['ymd'], ymin=0, ymax=keyword_plot['Keyword_Index'],
               color='darkblue', linewidth=0.8)
axes[0].set_title('(a) Keyword-Based Coordination Index', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Keyword Coordination Index', fontsize=12)
sbert_plot = daily_full.loc[np.isfinite(daily_full['SBERT_Index'])]
axes[1].vlines(sbert_plot['ymd'], ymin=0, ymax=sbert_plot['SBERT_Index'],
               color='darkred', linewidth=0.8)
axes[1].set_title('(b) SBERT-Based Coordination Index', fontsize=13, fontweight='bold')
axes[1].set_ylabel('SBERT Semantic Intensity', fontsize=12)
for ax in axes:
    format_axis(ax, daily_full['ymd'], frequency='daily')
fig.suptitle('International Monetary Policy Coordination Indices', fontsize=15, fontweight='bold')
fig.text(0.5, 0.01,
         'Notes: Shaded areas denote NBER recessions. Both indices apply a 90th-percentile threshold. '
         'Days without speeches are assigned zero.',
         ha='center', va='bottom', fontsize=9, wrap=True)
fig.tight_layout(rect=[0, 0.09, 1, 0.93], w_pad=2.5)
fig.savefig(f'{PREFIX}-Daily-Comparison.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{PREFIX}-Daily-Comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. MONTHLY FIGURE
fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharex=True)
axes[0].plot(monthly['date'], monthly['monthlylog_freqtrigger'], color='darkblue', linewidth=1.0)
axes[0].set_title('(a) Keyword-Based Coordination Index', fontsize=13, fontweight='bold')
axes[0].set_ylabel('Monthly Log Frequency Index', fontsize=12)
axes[1].plot(monthly['date'], monthly['SBERT_Log'], color='darkred', linewidth=1.0)
axes[1].set_title('(b) SBERT-Based Coordination Index', fontsize=13, fontweight='bold')
axes[1].set_ylabel('Monthly Log SBERT Index', fontsize=12)
for ax in axes:
    format_axis(ax, monthly['date'], frequency='monthly')
fig.suptitle('Monthly International Monetary Policy Coordination Indices', fontsize=15, fontweight='bold')
fig.text(0.5, 0.01,
         'Notes: Monthly indices are sums of daily indices. Shaded areas denote NBER recessions.',
         ha='center', va='bottom', fontsize=9)
fig.tight_layout(rect=[0, 0.09, 1, 0.93], w_pad=2.5)
fig.savefig(f'{PREFIX}-Monthly-Comparison.pdf', dpi=300, bbox_inches='tight')
fig.savefig(f'{PREFIX}-Monthly-Comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. SUMMARY STATISTICS
print('\nMONTHLY SUMMARY STATISTICS')
print(monthly[['monthlylog_freqtrigger', 'SBERT_Index', 'SBERT_Log', 'SBERT_Binary']].describe())
print('\nPositive keyword months:', int((monthly['monthlylog_freqtrigger'] > 0).sum()))
print('Positive SBERT months:', int((monthly['SBERT_Index'] > 0).sum()))
print('Positive SBERT days:', int((daily_full['SBERT_Binary'] > 0).sum()))
print('\nSBERT-02 COMPLETED')
