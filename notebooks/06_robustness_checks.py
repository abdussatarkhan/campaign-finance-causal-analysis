# %% [markdown]
# # Notebook 06: Sensitivity Analysis & Econometric Robustness Checks
# **Campaign Donation Influence & Legislative Voting Analyzer**
#
# This notebook conducts exhaustive sensitivity analyses across all causal estimators:
# 1. RDD Bandwidth Perturbation: Testing bandwidth multipliers from 0.5x to 2.0x
# 2. RDD Placebo Cutoff Falsification Tests at non-discontinuity margins ($c \in \{-0.10, -0.05, 0.05, 0.10\}$)
# 3. PSM Specification Checks: Caliper variations ($0.01$ to $0.10$) and replacement strategies
# 4. Clustered Logistic Regression Multi-Model Specification Comparison
# 5. Industry Heterogeneity: Subgroup treatment effects across economic sectors

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
from scripts.sensitivity_analysis import SensitivitySuite
from scripts.hypothesis_testing import ClusteredLogisticEstimator

config = load_config()
proc_dir = Path(config["paths"]["processed_data"])
img_dir = ensure_dir(repo_root / config["paths"]["figures_dir"])

# Load master analytical panel
df_panel = pd.read_parquet(proc_dir / "panel_analysis_master.parquet")
df_leg = df_panel.groupby("legislator_id").first().reset_index()

suite = SensitivitySuite(config)

# %% [markdown]
# ## 1. RDD Bandwidth Sensitivity
# Assessing whether the causal jump $\hat{\tau}$ is robust or sensitive to bandwidth selection.

# %%
df_bw = suite.rdd_bandwidth_sensitivity(df_leg["vote_margin"].values, df_leg["log_donations"].values)
print("RDD Bandwidth Sensitivity Results:")
print(df_bw)

plt.figure(figsize=(9, 5))
plt.errorbar(
    df_bw["Bandwidth_Value"],
    df_bw["Tau_Estimate"],
    yerr=1.96 * df_bw["Standard_Error"],
    fmt="o",
    color="#2b5c8f",
    ecolor="#e41a1c",
    elinewidth=2,
    capsize=5,
    label="Estimated Tau (95% CI)"
)
plt.axhline(0, color="gray", linestyle="--")
plt.title("RDD Treatment Effect Sensitivity across Bandwidth Windows", fontweight="bold")
plt.xlabel("Bandwidth Window (h)")
plt.ylabel("Estimated Causal Effect (Tau)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

fig_bw_sens = img_dir / "09_rdd_bandwidth_sensitivity.png"
plt.savefig(fig_bw_sens, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Bandwidth Sensitivity Plot to {fig_bw_sens}")

# %% [markdown]
# ## 2. RDD Placebo Cutoff Falsification Tests
# If the identification strategy is sound, artificial threshold shifts should yield null results ($p > 0.05$).

# %%
df_placebo = suite.rdd_placebo_cutoffs(df_leg["vote_margin"].values, df_leg["log_donations"].values)
print("\nRDD Placebo Cutoff Test Results:")
print(df_placebo)

# %% [markdown]
# ## 3. Propensity Score Matching Caliper & Replacement Sensitivity

# %%
df_psm = suite.psm_specification_sensitivity(df_panel)
print("\nPSM Sensitivity Results across Calipers & Replacement Policies:")
print(df_psm)

# %% [markdown]
# ## 4. Clustered Logistic Regression Multi-Model Comparison
# Evaluating parameter stability from uncontrolled baseline to fully conditioned specification.

# %%
estimator = ClusteredLogisticEstimator(cluster_col="legislator_id")
df_models = estimator.run_multi_specification_suite(df_panel)

print("\nEconometric Specification Ladder (Clustered Standard Errors):")
cols = ["Specification", "Donations_Beta", "Donations_SE", "Odds_Ratio", "AME", "P_Value", "Significance", "N_Obs"]
print(df_models[cols].to_string(index=False))

# Plotting coefficient stability across models
plt.figure(figsize=(10, 5))
y_pos = np.arange(len(df_models))
plt.errorbar(
    df_models["Donations_Beta"],
    y_pos,
    xerr=1.96 * df_models["Donations_SE"],
    fmt="s",
    color="#377eb8",
    ecolor="#e41a1c",
    elinewidth=2,
    capsize=5
)
plt.axvline(0, color="black", linestyle="--", alpha=0.7)
plt.yticks(y_pos, df_models["Specification"])
plt.title("Financial Influence Coefficient Stability Across Model Specifications", fontweight="bold")
plt.xlabel("Log Donations Coefficient (Beta) with 95% Clustered CI")
plt.grid(True, linestyle="--", alpha=0.5)

fig_model_ladder = img_dir / "10_econometric_specification_ladder.png"
plt.savefig(fig_model_ladder, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Model Comparison Ladder to {fig_model_ladder}")

# %% [markdown]
# ## 5. Industry Sector Heterogeneity

# %%
df_ind = suite.industry_heterogeneity(df_panel)
print("\nIndustry Subgroup Treatment Heterogeneity:")
print(df_ind)
