# %% [markdown]
# # Notebook 04: Regression Discontinuity Design (RDD) Analysis
# **Campaign Donation Influence & Legislative Voting Analyzer**
#
# This notebook estimates the causal impact of razor-thin election victories on campaign cash inflows
# and subsequent roll-call voting alignments using a Sharp Regression Discontinuity Design:
# 1. Running variable definition: General election two-party vote margin centered at threshold 0.0
# 2. McCrary (2008) density test to verify absence of sorting/manipulation at the boundary
# 3. Data-driven bandwidth selection using the Imbens-Kalyanaraman (2012) algorithm
# 4. Local linear and quadratic non-parametric regressions with triangular kernel weighting
# 5. Visual discontinuity plots with binned averages and confidence bands

# %%
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from scripts.utils import setup_logger, load_config, ensure_dir
from scripts.regression_discontinuity import McCraryDensityTest, LocalPolynomialRDD, ImbensKalyanaramanBandwidth

config = load_config()
proc_dir = Path(config["paths"]["processed_data"])
img_dir = ensure_dir(repo_root / config["paths"]["figures_dir"])

# Load master panel data
df_panel = pd.read_parquet(proc_dir / "panel_analysis_master.parquet")
# One row per legislator for candidate election margins
df_leg = df_panel.groupby("legislator_id").first().reset_index()
print(f"Loaded {len(df_leg)} congressional candidates / legislators.")

# %% [markdown]
# ## 1. McCrary (2008) Density Discontinuity Test
# The identifying assumption of RDD requires that candidates cannot precisely manipulate their vote margin around zero.
# We test for a discontinuous jump in candidate density at the election threshold.

# %%
density_test = McCraryDensityTest.run_test(df_leg["vote_margin"].values, cutoff=0.0)
print(f"McCrary Log-Difference (theta): {density_test['theta_log_diff']:.4f}")
print(f"Standard Error: {density_test['se_theta']:.4f}")
print(f"P-Value: {density_test['p_value']:.4f}")
print(f"Manipulation Detected: {density_test['manipulation_detected']}")

# Visualizing running variable distribution around cutoff
plt.figure(figsize=(9, 5))
sns.histplot(df_leg["vote_margin"], bins=40, kde=True, color="#2b5c8f", edgecolor="black")
plt.axvline(0, color="red", linestyle="--", linewidth=2, label="Election Cutoff (Margin = 0)")
plt.title("Distribution of Election Vote Margin (Running Variable)", fontweight="bold")
plt.xlabel("Two-Party General Election Vote Margin (Margin > 0: Winner)")
plt.ylabel("Candidate Frequency")
plt.legend()

fig_mccrary = img_dir / "06_rdd_running_variable_distribution.png"
plt.savefig(fig_mccrary, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved running variable plot to {fig_mccrary}")

# %% [markdown]
# ## 2. Optimal Bandwidth Selection (Imbens-Kalyanaraman)
# Bandwidth represents the bias-variance tradeoff: narrow windows reduce bias, wide windows increase precision.

# %%
h_opt = ImbensKalyanaramanBandwidth.calculate_bandwidth(
    df_leg["vote_margin"].values,
    df_leg["log_donations"].values,
    cutoff=0.0,
    kernel="triangular"
)
print(f"Optimal Bandwidth (h_IK): {h_opt:.4f}")

# %% [markdown]
# ## 3. Local Polynomial Regression: Local Linear vs Local Quadratic

# %%
# Local Linear Specification
rdd_linear = LocalPolynomialRDD(cutoff=0.0, polynomial_degree=1, kernel="triangular", bandwidth=h_opt)
res_linear = rdd_linear.fit(df_leg["vote_margin"].values, df_leg["log_donations"].values)

# Local Quadratic Specification
rdd_quad = LocalPolynomialRDD(cutoff=0.0, polynomial_degree=2, kernel="triangular", bandwidth=h_opt)
res_quad = rdd_quad.fit(df_leg["vote_margin"].values, df_leg["log_donations"].values)

print("\nRDD Estimation Comparison:")
comparison_df = pd.DataFrame([res_linear, res_quad], index=["Local Linear (p=1)", "Local Quadratic (p=2)"])
print(comparison_df[["treatment_effect_tau", "standard_error", "t_statistic", "p_value", "ci_lower", "ci_upper", "effective_sample_size"]])

# %% [markdown]
# ## 4. Regression Discontinuity Plot
# Binned scatter plot of observed data against fitted local regression polynomials on either side of threshold.

# %%
x = df_leg["vote_margin"].values
y = df_leg["log_donations"].values

# Restrict to window around cutoff
mask_window = np.abs(x) <= (h_opt * 1.5)
x_plot = x[mask_window]
y_plot = y[mask_window]

# Binning
bins = np.linspace(-h_opt * 1.5, h_opt * 1.5, 30)
df_binned = pd.DataFrame({"x": x_plot, "y": y_plot})
df_binned["bin"] = pd.cut(df_binned["x"], bins=bins)
bin_means = df_binned.groupby("bin", observed=True)[["x", "y"]].mean().dropna()

plt.figure(figsize=(10, 6))
# Plot binned points
plt.scatter(bin_means["x"], bin_means["y"], color="#1f77b4", s=60, alpha=0.9, label="Binned Means")

# Left polynomial fit (control)
x_left = np.linspace(-h_opt, -0.001, 100)
pred_left = res_linear["treatment_effect_tau"] * 0 + np.poly1d(np.polyfit(x[x < 0], y[x < 0], 1))(x_left)
plt.plot(x_left, pred_left, color="#d95f02", linewidth=2.5, label="Fitted Linear (Left)")

# Right polynomial fit (treated)
x_right = np.linspace(0.001, h_opt, 100)
pred_right = np.poly1d(np.polyfit(x[x >= 0], y[x >= 0], 1))(x_right)
plt.plot(x_right, pred_right, color="#2ca02c", linewidth=2.5, label="Fitted Linear (Right)")

plt.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Cutoff (Margin = 0)")
plt.title(f"Sharp RDD: Incumbency Victory Effect on Industry Campaign Receipts (h={h_opt:.3f})", fontweight="bold")
plt.xlabel("Two-Party Vote Margin at General Election")
plt.ylabel("ln(PAC Campaign Contributions)")
plt.legend(loc="upper left")

fig_rdd_plot = img_dir / "07_rdd_discontinuity_plot.png"
plt.savefig(fig_rdd_plot, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved RDD plot to {fig_rdd_plot}")
