"""
IPL Player Auction Value Predictor — Archishman Mittal
=======================================================
Predicts whether an IPL player is Overpriced, Fairly Valued, or Undervalued
at auction using performance metrics and ML regression + classification.

Combines:
  - Economics: market valuation, supply-demand, marginal utility of player roles
  - ML: Random Forest Regressor + Classifier (Scikit-learn)
  - Data: Player performance stats across IPL seasons

Models built:
  1. Price Predictor — regression model estimating fair auction price (₹ crore)
  2. Value Classifier — classifies player as Undervalued / Fair / Overpriced
  3. Feature importance — which stats drive auction price the most
  4. Role-wise value analysis — batters vs bowlers vs all-rounders
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── Dark theme ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

BG = '#0F1923'
PANEL = '#1A2535'
BLUE = '#1A56A0'
GOLD = '#F5A623'
GREEN = '#27AE60'
RED = '#E74C3C'
GRAY = '#AAB8C2'
GRID = '#2A3545'

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE REALISTIC PLAYER DATASET
# ═══════════════════════════════════════════════════════════════════════════════

players = [
    # name, role, matches, runs, avg, sr, wickets, economy, catches, age, seasons
    ("Virat Kohli",        "Batter",      220, 7200, 38.5, 131, 0,    0.0,  80,  34, 15),
    ("Rohit Sharma",       "Batter",      230, 6600, 30.2, 130, 0,    0.0,  70,  36, 15),
    ("MS Dhoni",           "WK-Batter",   230, 5000, 41.2, 135, 0,    0.0, 120,  41, 15),
    ("Jos Buttler",        "WK-Batter",   100, 3800, 42.0, 148, 0,    0.0,  55,  32, 6),
    ("Suryakumar Yadav",   "Batter",      120, 3900, 38.0, 170, 0,    0.0,  50,  32, 7),
    ("Faf du Plessis",     "Batter",      110, 3500, 35.0, 134, 0,    0.0,  45,  38, 8),
    ("Ruturaj Gaikwad",    "Batter",       80, 2600, 37.0, 136, 0,    0.0,  30,  26, 4),
    ("Ishan Kishan",       "WK-Batter",   100, 2800, 29.0, 136, 0,    0.0,  80,  24, 5),
    ("Shubman Gill",       "Batter",       90, 2900, 36.0, 132, 0,    0.0,  35,  23, 4),
    ("Devdutt Padikkal",   "Batter",       70, 1900, 28.0, 125, 0,    0.0,  25,  23, 3),
    ("Prithvi Shaw",       "Batter",       60, 1500, 25.0, 148, 0,    0.0,  20,  23, 4),
    ("Sanju Samson",       "WK-Batter",   130, 3600, 30.0, 138, 0,    0.0, 100,  28, 8),
    ("Glenn Maxwell",      "All-Rounder", 110, 2800, 28.0, 157, 30,   8.0,  55,  34, 8),
    ("Hardik Pandya",      "All-Rounder", 110, 2200, 30.0, 145, 55,   9.2,  50,  29, 8),
    ("Ravindra Jadeja",    "All-Rounder", 200, 2600, 26.0, 130, 130,  7.6,  90,  34, 14),
    ("Andre Russell",      "All-Rounder", 110, 2400, 30.0, 178, 90,   9.5,  40,  34, 9),
    ("Axar Patel",         "All-Rounder",  90, 1000, 18.0, 115, 80,   7.2,  40,  29, 7),
    ("Moeen Ali",          "All-Rounder",  70, 1200, 22.0, 148, 50,   8.0,  30,  36, 5),
    ("Washington Sundar",  "All-Rounder",  80,  900, 18.0, 118, 55,   7.8,  35,  23, 5),
    ("Venkatesh Iyer",     "All-Rounder",  60, 1400, 28.0, 138, 20,   9.0,  25,  27, 3),
    ("Jasprit Bumrah",     "Bowler",      100,   80,  5.0,  80, 140,  6.8,  35,  29, 8),
    ("Mohammed Siraj",     "Bowler",       80,   60,  4.0,  75, 100,  8.2,  25,  29, 5),
    ("Yuzvendra Chahal",   "Bowler",      130,   80,  5.0,  70, 170,  7.6,  40,  32, 10),
    ("Kagiso Rabada",      "Bowler",       80,   50,  4.0,  80, 100,  8.0,  30,  28, 5),
    ("Trent Boult",        "Bowler",       90,   40,  4.0,  70, 110,  8.4,  30,  33, 6),
    ("Harshal Patel",      "Bowler",       80,   60,  5.0,  90, 110,  9.0,  25,  32, 5),
    ("Arshdeep Singh",     "Bowler",       60,   30,  4.0,  65,  75,  8.6,  20,  24, 3),
    ("T Natarajan",        "Bowler",       60,   30,  3.0,  60,  75,  8.8,  15,  31, 4),
    ("Umran Malik",        "Bowler",       40,   20,  3.0,  55,  50,  9.5,  10,  23, 2),
    ("Varun Chakravarthy", "Bowler",       60,   30,  3.0,  60,  80,  7.8,  20,  31, 4),
    ("Liam Livingstone",   "All-Rounder",  60, 1600, 31.0, 168, 25,   9.0,  25,  29, 3),
    ("Shimron Hetmyer",    "Batter",       70, 1500, 28.0, 152, 0,    0.0,  20,  26, 4),
    ("Nicholas Pooran",    "WK-Batter",    70, 1700, 27.0, 148, 0,    0.0,  55,  27, 4),
    ("Aiden Markram",      "All-Rounder",  60, 1300, 26.0, 140, 20,   8.5,  25,  28, 3),
    ("Pat Cummins",        "Bowler",       60,  100,  8.0,  95,  80,  8.8,  20,  30, 4),
    ("Rashid Khan",        "Bowler",      100,  200,  9.0,  95, 130,  6.3,  45,  24, 6),
    ("Sunil Narine",       "All-Rounder", 170, 1800, 22.0, 160, 150,  6.7,  60,  34, 12),
    ("Kieron Pollard",     "All-Rounder", 170, 3200, 28.0, 147, 65,   9.8,  65,  35, 13),
    ("Mayank Agarwal",     "Batter",       80, 2100, 28.0, 140, 0,    0.0,  25,  31, 6),
    ("Nitish Rana",        "All-Rounder",  90, 2000, 25.0, 130, 15,   9.2,  35,  28, 5),
    ("Abhishek Sharma",    "All-Rounder",  50, 1200, 28.0, 158, 15,   9.5,  20,  22, 2),
    ("Shahrukh Khan",      "Batter",       50,  900, 22.0, 148, 0,    0.0,  15,  27, 3),
    ("Jonny Bairstow",     "WK-Batter",    50, 1600, 36.0, 142, 0,    0.0,  40,  33, 3),
    ("Devon Conway",       "WK-Batter",    50, 1500, 34.0, 130, 0,    0.0,  40,  31, 3),
    ("Dinesh Karthik",     "WK-Batter",   200, 4200, 27.0, 144, 0,    0.0, 160,  37, 15),
    ("David Warner",       "Batter",      160, 5800, 41.5, 142, 0,    0.0,  55,  36, 10),
    ("Kane Williamson",    "Batter",      100, 2700, 32.0, 120, 0,    0.0,  40,  32, 7),
    ("Mitchell Marsh",     "All-Rounder",  60, 1100, 25.0, 145, 25,   9.5,  25,  31, 4),
    ("Sam Billings",       "WK-Batter",    50,  900, 22.0, 132, 0,    0.0,  40,  31, 3),
    ("Wanindu Hasaranga",  "All-Rounder",  60,  500, 14.0, 125, 75,   7.5,  25,  25, 3),
]

rows = []
for p in players:
    name, role, matches, runs, avg, sr, wkts, eco, catches, age, seasons = p

    # Compute base price using a weighted economic formula
    # Batter value
    bat_val = (runs / max(matches, 1)) * 0.4 + avg * 0.3 + (sr - 100) * 0.1
    # Bowler value
    bowl_val = (wkts / max(matches, 1)) * 30 + max(0, 10 - eco) * 5
    # Fielding
    field_val = catches / max(matches, 1) * 5
    # Age penalty (peak 24-30, decline after)
    age_factor = 1.0 if 24 <= age <= 30 else (0.85 if age > 30 else 0.9)
    # Experience bonus
    exp_bonus = min(seasons * 0.3, 3.0)

    if role == 'Batter':
        base = (avg * 0.15) + ((sr - 100) * 0.06) + (runs / max(matches,1) * 0.5) + exp_bonus
    elif role == 'Bowler':
        base = (wkts / max(matches,1) * 8) + max(0, 10 - eco) * 1.2 + exp_bonus
    elif role == 'WK-Batter':
        base = (avg * 0.12) + ((sr - 100) * 0.05) + (catches / max(matches,1) * 3) + exp_bonus
    else:  # All-Rounder
        base = (avg * 0.08) + ((sr-100)*0.04) + (wkts/max(matches,1)*5) + max(0,10-eco)*0.6 + exp_bonus

    fair_price = round(base * age_factor * np.random.uniform(0.92, 1.08), 2)
    fair_price = max(0.5, min(fair_price, 20.0))

    # Actual auction price = fair price + market noise (demand, hype, bidding wars)
    noise = np.random.normal(0, fair_price * 0.25)
    actual_price = round(fair_price + noise, 2)
    actual_price = max(0.2, min(actual_price, 25.0))

    # Value label
    ratio = actual_price / fair_price
    if ratio > 1.25:
        value_label = 'Overpriced'
    elif ratio < 0.80:
        value_label = 'Undervalued'
    else:
        value_label = 'Fair Value'

    rows.append({
        'player': name, 'role': role, 'matches': matches,
        'runs': runs, 'batting_avg': avg, 'strike_rate': sr,
        'wickets': wkts, 'economy': eco, 'catches': catches,
        'age': age, 'seasons': seasons,
        'bat_contribution': round(bat_val, 2),
        'bowl_contribution': round(bowl_val, 2),
        'fair_price_cr': fair_price,
        'actual_price_cr': actual_price,
        'value_label': value_label,
    })

df = pd.DataFrame(rows)
print(f"Dataset: {len(df)} players")
print(df[['player','role','fair_price_cr','actual_price_cr','value_label']].head(10).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# ML MODEL 1 — PRICE REGRESSION (predict fair price)
# ═══════════════════════════════════════════════════════════════════════════════
le = LabelEncoder()
df['role_enc'] = le.fit_transform(df['role'])

features = ['role_enc','matches','runs','batting_avg','strike_rate',
            'wickets','economy','catches','age','seasons']
X = df[features]
y_price = df['fair_price_cr']
y_label = df['value_label']

X_train, X_test, yp_train, yp_test = train_test_split(X, y_price, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=6)
reg_model.fit(X_train, yp_train)
yp_pred = reg_model.predict(X_test)
mae = mean_absolute_error(yp_test, yp_pred)
r2 = r2_score(yp_test, yp_pred)
print(f"\n💰 Price Regressor — MAE: ₹{mae:.2f} Cr | R²: {r2:.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# ML MODEL 2 — VALUE CLASSIFIER (Undervalued / Fair / Overpriced)
# ═══════════════════════════════════════════════════════════════════════════════
le2 = LabelEncoder()
y_enc = le2.fit_transform(y_label)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

clf_model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=6)
clf_model.fit(Xc_train, yc_train)
yc_pred = clf_model.predict(Xc_test)
acc = accuracy_score(yc_test, yc_pred)
cv_scores = cross_val_score(clf_model, X, y_enc, cv=5)
print(f"🏷️  Value Classifier — Accuracy: {acc*100:.1f}% | CV Mean: {cv_scores.mean()*100:.1f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — MAIN ANALYTICS DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

# ── 1A: Feature Importance ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor(PANEL)

importances = reg_model.feature_importances_
feat_names = ['Role','Matches','Runs','Batting Avg','Strike Rate',
              'Wickets','Economy','Catches','Age','Seasons']
sorted_idx = np.argsort(importances)
colors_fi = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(feat_names)))

bars1 = ax1.barh([feat_names[i] for i in sorted_idx],
                  importances[sorted_idx], color=colors_fi, edgecolor='none', height=0.65)
for bar, val in zip(bars1, importances[sorted_idx]):
    ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f'{val:.3f}', va='center', fontsize=9, color='white', fontweight='bold')

ax1.set_xlabel('Feature Importance Score', color=GRAY, fontsize=10)
ax1.set_title(f'What Drives Auction Price?\n(R² = {r2:.2f}, MAE = ₹{mae:.2f} Cr)',
              color='white', fontsize=12, fontweight='bold', pad=10)
ax1.tick_params(colors=GRAY, labelsize=9)
ax1.spines['bottom'].set_color(GRAY)
ax1.spines['left'].set_color(GRAY)
ax1.grid(axis='x', color=GRID, linestyle='--', alpha=0.5)

# ── 1B: Fair Price vs Actual Price scatter ────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor(PANEL)

color_map = {'Undervalued': GREEN, 'Fair Value': GOLD, 'Overpriced': RED}
for label, grp in df.groupby('value_label'):
    ax2.scatter(grp['fair_price_cr'], grp['actual_price_cr'],
                c=color_map[label], label=label, s=80, alpha=0.85, edgecolors='none')

max_val = max(df['fair_price_cr'].max(), df['actual_price_cr'].max()) + 1
ax2.plot([0, max_val], [0, max_val], '--', color='white', alpha=0.4, linewidth=1.5, label='Perfect Value Line')
ax2.plot([0, max_val], [0, max_val * 1.25], '--', color=RED, alpha=0.3, linewidth=1, label='+25% (Overpriced zone)')
ax2.plot([0, max_val], [0, max_val * 0.80], '--', color=GREEN, alpha=0.3, linewidth=1, label='-20% (Undervalued zone)')

# Annotate notable players
notable = ['Virat Kohli', 'Jasprit Bumrah', 'MS Dhoni', 'Rashid Khan', 'Suryakumar Yadav']
for _, row in df[df['player'].isin(notable)].iterrows():
    ax2.annotate(row['player'].split()[0], (row['fair_price_cr'], row['actual_price_cr']),
                 fontsize=7.5, color='white', alpha=0.9,
                 xytext=(4, 4), textcoords='offset points')

ax2.set_xlabel('Fair Price (₹ Crore)', color=GRAY, fontsize=10)
ax2.set_ylabel('Actual Auction Price (₹ Crore)', color=GRAY, fontsize=10)
ax2.set_title('Fair Value vs Actual Auction Price\n(Each dot = 1 player)',
              color='white', fontsize=12, fontweight='bold', pad=10)
ax2.legend(fontsize=8, facecolor=PANEL, labelcolor='white', edgecolor=GRID, loc='upper left')
ax2.tick_params(colors=GRAY, labelsize=9)
ax2.spines['bottom'].set_color(GRAY)
ax2.spines['left'].set_color(GRAY)
ax2.grid(color=GRID, linestyle='--', alpha=0.4)

# ── 1C: Value Distribution by Role ────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.set_facecolor(PANEL)

role_value = df.groupby(['role', 'value_label']).size().unstack(fill_value=0)
roles = role_value.index.tolist()
x = np.arange(len(roles))
width = 0.25

for i, (label, color) in enumerate([('Undervalued', GREEN), ('Fair Value', GOLD), ('Overpriced', RED)]):
    if label in role_value.columns:
        bars3 = ax3.bar(x + i * width, role_value[label], width, label=label,
                        color=color, alpha=0.85, edgecolor='none')

ax3.set_xticks(x + width)
ax3.set_xticklabels(roles, color=GRAY, fontsize=9)
ax3.set_ylabel('Number of Players', color=GRAY, fontsize=10)
ax3.set_title('Value Classification by Player Role\n(Which roles are most mispriced at auction?)',
              color='white', fontsize=12, fontweight='bold', pad=10)
ax3.legend(fontsize=9, facecolor=PANEL, labelcolor='white', edgecolor=GRID)
ax3.tick_params(colors=GRAY, labelsize=9)
ax3.spines['bottom'].set_color(GRAY)
ax3.spines['left'].set_color(GRAY)
ax3.grid(axis='y', color=GRID, linestyle='--', alpha=0.5)

# ── 1D: Age vs Fair Price ──────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.set_facecolor(PANEL)

for label, grp in df.groupby('value_label'):
    ax4.scatter(grp['age'], grp['fair_price_cr'],
                c=color_map[label], s=70, alpha=0.8, label=label, edgecolors='none')

# Trend line
z = np.polyfit(df['age'], df['fair_price_cr'], 2)
p = np.poly1d(z)
age_range = np.linspace(df['age'].min(), df['age'].max(), 100)
ax4.plot(age_range, p(age_range), '--', color=GOLD, linewidth=2, alpha=0.7, label='Trend')

ax4.set_xlabel('Player Age', color=GRAY, fontsize=10)
ax4.set_ylabel('Fair Price (₹ Crore)', color=GRAY, fontsize=10)
ax4.set_title('Age vs Fair Auction Value\n(Peak value window: 24–30 years)',
              color='white', fontsize=12, fontweight='bold', pad=10)
ax4.axvspan(24, 30, alpha=0.08, color=GREEN, label='Peak window')
ax4.legend(fontsize=8, facecolor=PANEL, labelcolor='white', edgecolor=GRID)
ax4.tick_params(colors=GRAY, labelsize=9)
ax4.spines['bottom'].set_color(GRAY)
ax4.spines['left'].set_color(GRAY)
ax4.grid(color=GRID, linestyle='--', alpha=0.4)

fig.suptitle('IPL Auction Value Predictor — Market Analysis Dashboard',
             color='white', fontsize=16, fontweight='bold', y=0.98)

plt.savefig('ipl_auction_dashboard.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("\n✅ Saved: ipl_auction_dashboard.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — TOP BARGAINS & MOST OVERPRICED
# ═══════════════════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(16, 7))
fig2.patch.set_facecolor(BG)

df['value_gap'] = df['actual_price_cr'] - df['fair_price_cr']
df['value_gap_pct'] = ((df['actual_price_cr'] - df['fair_price_cr']) / df['fair_price_cr'] * 100).round(1)

# Top 10 undervalued
ax5 = axes[0]
ax5.set_facecolor(PANEL)
undervalued = df[df['value_label'] == 'Undervalued'].sort_values('value_gap').head(10)
bars5 = ax5.barh(undervalued['player'], abs(undervalued['value_gap']),
                 color=GREEN, edgecolor='none', height=0.65)
for bar, (_, row) in zip(bars5, undervalued.iterrows()):
    ax5.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f'₹{abs(row["value_gap"]):.1f}Cr below fair value',
             va='center', fontsize=8.5, color='white')

ax5.set_xlabel('Underpriced by (₹ Crore)', color=GRAY, fontsize=10)
ax5.set_title('🟢 Best Bargains at Auction\n(Most Undervalued Players)',
              color='white', fontsize=13, fontweight='bold', pad=10)
ax5.tick_params(colors=GRAY, labelsize=9)
ax5.spines['bottom'].set_color(GRAY)
ax5.spines['left'].set_color(GRAY)
ax5.grid(axis='x', color=GRID, linestyle='--', alpha=0.5)

# Top 10 overpriced
ax6 = axes[1]
ax6.set_facecolor(PANEL)
overpriced = df[df['value_label'] == 'Overpriced'].sort_values('value_gap', ascending=False).head(10)
bars6 = ax6.barh(overpriced['player'], overpriced['value_gap'],
                 color=RED, edgecolor='none', height=0.65)
for bar, (_, row) in zip(bars6, overpriced.iterrows()):
    ax6.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
             f'₹{row["value_gap"]:.1f}Cr above fair value',
             va='center', fontsize=8.5, color='white')

ax6.set_xlabel('Overpriced by (₹ Crore)', color=GRAY, fontsize=10)
ax6.set_title('🔴 Most Overpriced at Auction\n(Highest Price Premium Paid)',
              color='white', fontsize=13, fontweight='bold', pad=10)
ax6.tick_params(colors=GRAY, labelsize=9)
ax6.spines['bottom'].set_color(GRAY)
ax6.spines['left'].set_color(GRAY)
ax6.grid(axis='x', color=GRID, linestyle='--', alpha=0.5)

fig2.suptitle('IPL Auction Value Predictor — Bargain Hunter & Overpay Analysis',
              color='white', fontsize=15, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('ipl_auction_bargains.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("✅ Saved: ipl_auction_bargains.png")

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("IPL AUCTION VALUE PREDICTOR — SUMMARY")
print("="*70)
print(f"  Total Players Analysed : {len(df)}")
print(f"  Undervalued            : {(df['value_label']=='Undervalued').sum()}")
print(f"  Fair Value             : {(df['value_label']=='Fair Value').sum()}")
print(f"  Overpriced             : {(df['value_label']=='Overpriced').sum()}")
print(f"  Price Regressor R²     : {r2:.2f}")
print(f"  Value Classifier Acc   : {acc*100:.1f}%  (5-fold CV: {cv_scores.mean()*100:.1f}%)")
print(f"\n  Top 3 Undervalued Picks:")
top_bargains = df[df['value_label']=='Undervalued'].sort_values('value_gap').head(3)
for _, r in top_bargains.iterrows():
    print(f"    • {r['player']:25s} Fair: ₹{r['fair_price_cr']:.1f}Cr  Actual: ₹{r['actual_price_cr']:.1f}Cr")
print(f"\n  Top 3 Overpriced Players:")
top_over = df[df['value_label']=='Overpriced'].sort_values('value_gap', ascending=False).head(3)
for _, r in top_over.iterrows():
    print(f"    • {r['player']:25s} Fair: ₹{r['fair_price_cr']:.1f}Cr  Actual: ₹{r['actual_price_cr']:.1f}Cr")
print("="*70)
