# %% [markdown]
# # Notebook 03: Propensity Score Matching (PSM) & Balance Diagnostics
# **Campaign Donation Influence & Legislative Voting Analyzer**
#
# This notebook implements observational causal adjustment via Propensity Score Matching:
# 1. Estimation of propensity scores $P(T_i = 1 | X_i)$ using logistic regression
# 2. Common support verification and overlap inspection
# 3. 1:1 Nearest-Neighbor matching with strict caliper constraint
# 4. Standardized Mean Difference (SMD) balance diagnostics and Love Plot visualization
# 5. Average Treatment Effect on the Treated (ATT) calculation on voting outcomes

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
from scripts.propensity_matching import PropensityScoreMatcher

config = load_config()
proc_dir = Path(config["paths"]["processed_data"])
img_dir = ensure_dir(repo_root / config["paths"]["figures_dir"])

# Load master analytical panel
panel_file = proc_dir / "panel_analysis_master.parquet"
df_panel = pd.read_parquet(panel_file)
print(f"Loaded master panel with {len(df_panel)} records.")

# %% [markdown]
# ## 1. Propensity Score Estimation
# We model selection into high-donation status as a function of party, ideology (DW-NOMINATE),
# tenure, and district economic/demographic profile.

# %%
psm = PropensityScoreMatcher(
    treatment_col="high_donation_treatment",
    outcome_col="voted_pro_industry",
    caliper=config["causal_inference"]["propensity_score_matching"]["caliper"],
    replace=config["causal_inference"]["propensity_score_matching"]["replace"]
)

df_scored = psm.fit_propensity_scores(df_panel)
print("Propensity score summary:")
print(df_scored.groupby("high_donation_treatment")["propensity_score"].describe())

# %% [markdown]
# ## 2. Common Support / Propensity Score Overlap Plot
# Positivity assumption verification: ensure overlap in propensity scores between treated and control units.

# %%
plt.figure(figsize=(9, 5))
sns.kdeplot(
    data=df_scored[df_scored["high_donation_treatment"] == 1]["propensity_score"],
    label="High Donor Recipient (Treated)",
    color="#d95f02",
    fill=True,
    alpha=0.4
)
sns.kdeplot(
    data=df_scored[df_scored["high_donation_treatment"] == 0]["propensity_score"],
    label="Low / Zero Donor Recipient (Control)",
    color="#4a7bb0",
    fill=True,
    alpha=0.4
)
plt.title("Propensity Score Common Support Overlap", fontweight="bold")
plt.xlabel("Estimated Propensity Score P(Treated | Covariates)")
plt.ylabel("Density")
plt.legend()

fig_ps_overlap = img_dir / "04_propensity_score_overlap.png"
plt.savefig(fig_ps_overlap, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved overlap plot to {fig_ps_overlap}")

# %% [markdown]
# ## 3. Nearest-Neighbor Matching with Caliper Restriction

# %%
df_matched, balance_df = psm.match_nearest_neighbors(df_scored)
print(f"Original Treated Units: {(df_scored['high_donation_treatment'] == 1).sum()}")
print(f"Matched Units in Clean Sample: {len(df_matched)}")

# %% [markdown]
# ## 4. Balance Diagnostics: Love Plot
# Standardized Mean Differences before and after matching. |SMD| < 0.10 denotes robust balance.

# %%
plt.figure(figsize=(10, 6))

y_positions = np.arange(len(balance_df))
plt.scatter(balance_df["smd_pre_match"].abs(), y_positions, color="#e41a1c", label="Pre-Match |SMD|", s=80, marker="o")
plt.scatter(balance_df["smd_post_match"].abs(), y_positions, color="#377eb8", label="Post-Match |SMD|", s=90, marker="D")

for idx, row in balance_df.iterrows():
    plt.plot([abs(row["smd_pre_match"]), abs(row["smd_post_match"])], [idx, idx], color="gray", linestyle=":", alpha=0.8)

plt.axvline(0.10, color="darkgreen", linestyle="--", label="Balance Threshold (|SMD| = 0.10)")
plt.yticks(y_positions, balance_df["covariate"])
plt.title("Covariate Balance Diagnostics (Love Plot)", fontweight="bold")
plt.xlabel("Absolute Standardized Mean Difference (|SMD|)")
plt.legend(loc="upper right")
plt.grid(True, linestyle="--", alpha=0.6)

fig_love_plot = img_dir / "05_psm_love_plot.png"
plt.savefig(fig_love_plot, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved Love Plot to {fig_love_plot}")

print("\nBalance Diagnostics Summary Table:")
print(balance_df[["covariate", "smd_pre_match", "smd_post_match", "smd_reduction_pct", "balanced_status"]].to_string(index=False))

# %% [markdown]
# ## 5. Treatment Effect on the Treated (ATT)

# %%
att_results = psm.estimate_att(df_matched)
print("\n" + "="*50)
print("AVERAGE TREATMENT EFFECT ON THE TREATED (ATT)")
print("="*50)
for k, v in att_results.items():
    print(f"  {k:30s}: {v}")
