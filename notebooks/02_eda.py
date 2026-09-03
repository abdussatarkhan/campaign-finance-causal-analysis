# %% [markdown]
# # Notebook 02: Exploratory Data Analysis (EDA) & Stylized Facts
# **Campaign Donation Influence & Legislative Voting Analyzer**
#
# This notebook examines:
# 1. Distribution of PAC campaign contributions across industries (heavy right skew, log-normality)
# 2. Partisan and ideological differences in industry financing (DW-NOMINATE vs Contributions)
# 3. Congressional roll-call voting patterns across target industries
# 4. Temporal trends across election cycles (2018–2022)

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

config = load_config()
proc_dir = Path(config["paths"]["processed_data"])
img_dir = ensure_dir(repo_root / config["paths"]["figures_dir"])

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

# Load integrated panel data or build if not present
panel_file = proc_dir / "panel_analysis_master.parquet"
if not panel_file.exists():
    from scripts.data_integration import PanelDataIntegrator
    integrator = PanelDataIntegrator(config)
    df_panel = integrator.build_integrated_panel()
else:
    df_panel = pd.read_parquet(panel_file)

print(f"Loaded Master Analytical Panel: {df_panel.shape[0]} rows, {df_panel.shape[1]} columns")

# %% [markdown]
# ## 1. Campaign Contribution Distributions across Industries
# Political donations are heavily right-skewed with extreme outliers (PAC mega-donations).
# We examine both linear and log-transformed funding distributions.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Untransformed PAC Amount Distribution
sns.histplot(
    data=df_panel,
    x="total_industry_donations",
    hue="industry_code",
    bins=30,
    element="step",
    common_norm=False,
    ax=axes[0]
)
axes[0].set_title("Distribution of Total Industry PAC Donations (Raw USD)", fontweight="bold")
axes[0].set_xlabel("Donation Amount ($)")
axes[0].set_ylabel("Vote Record Frequency")

# Log-Transformed PAC Amount
sns.kdeplot(
    data=df_panel,
    x="log_donations",
    hue="industry_code",
    fill=True,
    common_norm=False,
    alpha=0.3,
    ax=axes[1]
)
axes[1].set_title("Log-Transformed Industry Donations: ln(1 + Donations)", fontweight="bold")
axes[1].set_xlabel("ln(1 + Amount)")
axes[1].set_ylabel("Density")

fig_path_1 = img_dir / "01_contribution_distributions.png"
plt.savefig(fig_path_1, dpi=300)
plt.close()
print(f"Saved distribution figure to {fig_path_1}")

# %% [markdown]
# ## 2. Ideology (DW-NOMINATE) vs. PAC Cash Flows
# Political scientists model legislator ideology along the 1st Dimension of DW-NOMINATE:
# Negative values represent liberal Democrats; positive values represent conservative Republicans.

# %%
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df_panel.drop_duplicates(subset=["legislator_id", "industry_code"]),
    x="dw_nominate_dim1",
    y="log_donations",
    hue="industry_code",
    style="party",
    alpha=0.7,
    s=70
)
plt.axvline(0, color="gray", linestyle="--", alpha=0.7, label="Ideological Center")
plt.title("DW-NOMINATE Dimension 1 vs. Industry PAC Campaign Receipts", fontweight="bold")
plt.xlabel("DW-NOMINATE Dim 1 (-1: Liberal, +1: Conservative)")
plt.ylabel("ln(Industry Donations)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

fig_path_2 = img_dir / "02_ideology_vs_donations.png"
plt.savefig(fig_path_2, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved ideology scatter plot to {fig_path_2}")

# %% [markdown]
# ## 3. Pro-Industry Legislative Voting Rates by Party & Treatment Group
# Comparing voting alignment with industry preferences across high-donation recipients vs control group.

# %%
voting_summary = df_panel.groupby(["industry_code", "party", "high_donation_treatment"])["voted_pro_industry"].agg(["count", "mean"]).reset_index()
voting_summary.rename(columns={"mean": "pro_industry_vote_share", "count": "total_votes"}, inplace=True)
voting_summary["treatment_group"] = voting_summary["high_donation_treatment"].map({1: "High Donor Recipient", 0: "Baseline / Low Donor"})

plt.figure(figsize=(10, 6))
sns.barplot(
    data=voting_summary,
    x="industry_code",
    y="pro_industry_vote_share",
    hue="treatment_group",
    palette=["#4a7bb0", "#d95f02"]
)
plt.title("Pro-Industry Voting Probability: High-Donation vs Control Legislators", fontweight="bold")
plt.xlabel("Industry Code")
plt.ylabel("Pro-Industry Vote Probability")
plt.ylim(0, 1.0)

fig_path_3 = img_dir / "03_pro_industry_voting_by_treatment.png"
plt.savefig(fig_path_3, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved voting probability chart to {fig_path_3}")

# %% [markdown]
# ## 4. Summary Statistics Table

# %%
summary_tbl = df_panel.groupby("industry_code").agg(
    n_votes=("vote_record_id", "count"),
    n_legislators=("legislator_id", "nunique"),
    mean_donation=("total_industry_donations", "mean"),
    median_donation=("total_industry_donations", "median"),
    std_donation=("total_industry_donations", "std"),
    pro_industry_rate=("voted_pro_industry", "mean")
).reset_index()

print("Summary Statistics by Industrial Sector:")
print(summary_tbl.to_string(index=False))
