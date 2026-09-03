# %% [markdown]
# # Notebook 05: Difference-in-Differences (DiD) & Event Study Analysis
# **Campaign Donation Influence & Legislative Voting Analyzer**
#
# This notebook implements longitudinal Difference-in-Differences to evaluate whether sudden PAC donation shocks
# (> 2 standard deviations above legislator historical baseline) cause shifts in legislative roll-call votes:
# 1. Identification of contribution spikes across congressional cycles
# 2. Static Two-Way Fixed Effects (TWFE) regression with legislator and year fixed effects
# 3. Dynamic event study specification estimating relative period trajectories ($t - t^* \in [-2, +2]$)
# 4. Statistical hypothesis test of the Parallel Trends Assumption (Wald F-Test on pre-treatment leads)
# 5. Publication-quality event study plot with error bands

# %%
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from scripts.utils import setup_logger, load_config, ensure_dir
from scripts.difference_in_differences import DifferenceInDifferencesEstimator, SpikeDetector

config = load_config()
proc_dir = Path(config["paths"]["processed_data"])
img_dir = ensure_dir(repo_root / config["paths"]["figures_dir"])

# Load integrated analytical panel
df_panel = pd.read_parquet(proc_dir / "panel_analysis_master.parquet")
print(f"Loaded {len(df_panel)} roll-call records from master panel.")

# %% [markdown]
# ## 1. Static Difference-in-Differences Estimation
# Outcome: $Y_{it} = \text{voted\_pro\_industry}$
# Regressors: $\text{Treated}_i \times \text{Post}_t$, Unit and Time Fixed Effects, Cluster-Robust SEs.

# %%
estimator = DifferenceInDifferencesEstimator(config)
static_results = estimator.estimate_static_did(df_panel)

print("\n" + "="*50)
print("STATIC DIFFERENCE-IN-DIFFERENCES ESTIMATES")
print("="*50)
for k, v in static_results.items():
    print(f"  {k:25s}: {v}")

# %% [markdown]
# ## 2. Dynamic Event Study Specification
# We estimate coefficients for relative event quarters leading up to and following the donation spike:
# $Y_{it} = \alpha_i + \lambda_t + \sum_{k \neq -1} \beta_k \cdot (\text{Treated}_i \times \mathbf{1}(t = k)) + \epsilon_{it}$
#
# The period immediately preceding the shock ($k = -1$) serves as the benchmark normalized to zero.

# %%
df_event_study, parallel_trends_test = estimator.estimate_event_study(df_panel, n_leads=2, n_lags=2)

print("\nEvent Study Trajectory:")
print(df_event_study[["relative_period", "coefficient", "standard_error", "ci_lower", "ci_upper", "period_type"]])

print("\nParallel Trends Joint Wald F-Test:")
print(f"  F-Statistic: {parallel_trends_test['wald_f_stat']:.4f}")
print(f"  P-Value:     {parallel_trends_test['wald_p_value']:.4f}")
print(f"  Result:      {'PASSED (Parallel trends hold)' if parallel_trends_test['parallel_trends_satisfied'] else 'VIOLATED'}")

# %% [markdown]
# ## 3. Event Study Plot
# Plotting estimated coefficients $\hat{\beta}_k$ along with 95% confidence intervals across event time.

# %%
plt.figure(figsize=(10, 6))

periods = df_event_study["relative_period"].values
coefs = df_event_study["coefficient"].values
ci_lowers = df_event_study["ci_lower"].values
ci_uppers = df_event_study["ci_upper"].values

# Plot point estimates
plt.plot(periods, coefs, color="#1f77b4", marker="o", linewidth=2, markersize=7, label="Event Study Treatment Estimate")
plt.fill_between(periods, ci_lowers, ci_uppers, color="#1f77b4", alpha=0.2, label="95% Clustered CI")

# Plot zero reference and event threshold
plt.axhline(0, color="gray", linestyle="--", alpha=0.7)
plt.axvline(-0.5, color="red", linestyle=":", linewidth=2, label="Donation Spike Timing (t = 0)")

plt.title("Dynamic Event Study: Impact of Campaign Donation Shocks on Legislative Voting", fontweight="bold")
plt.xlabel("Event Time Relative to PAC Donation Shock (Years)")
plt.ylabel("Change in Probability of Pro-Industry Vote")
plt.xticks(periods, [f"t {p:+d}" if p != 0 else "t = 0 (Shock)" for p in periods])
plt.legend(loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)

fig_event_study = img_dir / "08_did_event_study.png"
plt.savefig(fig_event_study, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Event Study Plot to {fig_event_study}")
