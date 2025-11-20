import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# Configuration
# ==========================================
LOG_FILE_1 = 'model_night/ocam_rl_log_night1.csv'  # File 1
LOG_FILE_2 = 'model_night/ocam_rl_log_night2.csv'  # File 2

LABEL_1 = "Run 1 (Night1)"
LABEL_2 = "Run 2 (Night2)"

# Phase Boundaries
PHASE_1_END = 460
PHASE_2_END = 745
# ==========================================

def load_and_process_data(filename):
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None

    # Data Preprocessing
    df['reward_ma'] = df['reward'].rolling(window=50).mean()

    # Blind Detection Logic
    max_exposure = 330
    blind_threshold = 30
    exposure_limit_threshold = max_exposure * 0.95

    df['real_fail_blind'] = (
        ((df['mu'] < blind_threshold) & (df['exp_100us'] < exposure_limit_threshold)) | 
        ((df['sat_hi'] + df['sat_lo']) > 0.15)
    )
    df['cumulative_blind'] = df['real_fail_blind'].cumsum()
    
    return df

# Load files
df1 = load_and_process_data(LOG_FILE_1)
df2 = load_and_process_data(LOG_FILE_2)

# Generate dummy data if file not found
if df1 is None:
    df1 = pd.DataFrame({'step': range(1200), 'reward': np.random.randn(1200), 'mu': np.random.randint(50, 150, 1200), 'exp_100us': 330, 'sat_hi':0, 'sat_lo':0})
    df1['reward_ma'] = df1['reward'].rolling(50).mean()
    df1['cumulative_blind'] = np.cumsum(np.random.randint(0,2,1200))
if df2 is None:
    df2 = pd.DataFrame({'step': range(1200), 'reward': np.random.randn(1200)-2, 'mu': np.random.randint(20, 100, 1200), 'exp_100us': 330, 'sat_hi':0, 'sat_lo':0})
    df2['reward_ma'] = df2['reward'].rolling(50).mean()
    df2['cumulative_blind'] = np.cumsum(np.random.randint(0,5,1200))


# Plotting
plt.figure(figsize=(12, 12))

def draw_phase_lines(ax, y_min, y_max):
    ax.axvline(x=PHASE_1_END, color='black', linestyle='--', alpha=0.4, linewidth=1)
    ax.axvline(x=PHASE_2_END, color='black', linestyle='--', alpha=0.4, linewidth=1)
    
    text_y = y_min + (y_max - y_min) * 0.95
    ax.text(PHASE_1_END/2, text_y, 'Phase 1\n(3s)', ha='center', fontsize=9, fontweight='bold', color='#555')
    ax.text((PHASE_1_END+PHASE_2_END)/2, text_y, 'Phase 2\n(2s)', ha='center', fontsize=9, fontweight='bold', color='#555')
    ax.text(PHASE_2_END + 150, text_y, 'Phase 3\n(1s)', ha='center', fontsize=9, fontweight='bold', color='#555')

# [Plot 1] Learning Curve Comparison
ax1 = plt.subplot(3, 1, 1)
ax1.plot(df1['step'], df1['reward_ma'], color='green', linewidth=2, label=f'{LABEL_1} Reward')
ax1.plot(df2['step'], df2['reward_ma'], color='blue', linewidth=2, alpha=0.7, label=f'{LABEL_2} Reward')

all_reward_min = min(df1['reward_ma'].min(), df2['reward_ma'].min())
all_reward_max = max(df1['reward_ma'].max(), df2['reward_ma'].max())
draw_phase_lines(ax1, all_reward_min, all_reward_max)

plt.title('1. Learning Stability Comparison', fontsize=14)
plt.ylabel('Reward (Smoothed)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')


# [Plot 2] Performance: Cumulative Control Failures
ax2 = plt.subplot(3, 1, 2)
ax2.plot(df1['step'], df1['cumulative_blind'], color='green', linewidth=2.5, label=f'{LABEL_1} Failures')
ax2.plot(df2['step'], df2['cumulative_blind'], color='blue', linewidth=2.5, linestyle='--', label=f'{LABEL_2} Failures')

all_fail_max = max(df1['cumulative_blind'].max(), df2['cumulative_blind'].max())
draw_phase_lines(ax2, 0, all_fail_max)

plt.title('2. Performance: Cumulative Control Failures', fontsize=14)
plt.ylabel('Failure Count', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')

plt.text(df1['step'].iloc[-1], df1['cumulative_blind'].iloc[-1], 
         f" {LABEL_1}: {df1['cumulative_blind'].iloc[-1]}", va='bottom', color='green', fontweight='bold')
plt.text(df2['step'].iloc[-1], df2['cumulative_blind'].iloc[-1], 
         f" {LABEL_2}: {df2['cumulative_blind'].iloc[-1]}", va='top', color='blue', fontweight='bold')


# [Plot 3] Brightness Tracking Comparison
ax3 = plt.subplot(3, 1, 3)

ax3.plot(df1['step'], df1['mu'], label=f'{LABEL_1} Brightness', color='green', linewidth=1, alpha=0.6)
ax3.plot(df2['step'], df2['mu'], label=f'{LABEL_2} Brightness', color='blue', linewidth=1, alpha=0.6)

ax3.axhline(y=100, color='gray', linestyle='--', linewidth=2, label='Target (100)')
ax3.axhline(y=27, color='red', linestyle=':', linewidth=2, label='HW Limit')
ax3.axhspan(0, 27, color='red', alpha=0.05)

draw_phase_lines(ax3, 0, 260)

plt.title('3. Brightness Tracking Comparison', fontsize=14)
plt.xlabel('Steps', fontsize=12)
plt.ylabel('Brightness (0-255)', fontsize=12)
plt.ylim(0, 260)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right', fontsize=10, ncol=2)

plt.tight_layout()
plt.show()