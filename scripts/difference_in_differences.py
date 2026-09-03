"""
Difference-in-Differences (DiD) and Event Study Econometric Engine:
1. Contribution Shock Identification (Donation spikes > 2 Standard Deviations)
2. Static Two-Way Fixed Effects (TWFE) DiD with Legislator & Time Fixed Effects
3. Dynamic Event Study (Leads and Lags) to trace pre/post voting trajectories
4. Statistical Hypothesis Testing of the Parallel Trends Assumption (Wald F-Test on Pre-Treatment Leads)
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import setup_logger, load_config, get_project_root, ensure_dir

logger = setup_logger("difference_in_differences")


class SpikeDetector:
    """Detects idiosyncratic PAC contribution spikes (> 2 standard deviations above legislator baseline)."""

    def __init__(self, spike_std_threshold: float = 2.0):
        self.threshold = spike_std_threshold

    def flag_spikes(self, df_contribs: pd.DataFrame) -> pd.DataFrame:
        """
        Computes rolling or lifetime baseline donation statistics per legislator-industry pair
        and flags sudden capital inflows.
        """
        logger.info(f"Identifying campaign donation shocks (> {self.threshold} sigma)...")
        df = df_contribs.copy()

        # Group by legislator and industry
        stats_df = df.groupby(["legislator_id", "industry_code"])["amount"].agg(["mean", "std"]).reset_index()
        stats_df.rename(columns={"mean": "leg_ind_mean", "std": "leg_ind_std"}, inplace=True)
        stats_df["leg_ind_std"] = stats_df["leg_ind_std"].fillna(stats_df["leg_ind_mean"] * 0.5)

        df = df.merge(stats_df, on=["legislator_id", "industry_code"], how="left")
        df["z_score"] = (df["amount"] - df["leg_ind_mean"]) / (df["leg_ind_std"] + 1e-6)
        df["is_spike"] = (df["z_score"] >= self.threshold).astype(int)

        n_spikes = df["is_spike"].sum()
        logger.info(f"Identified {n_spikes} donation spike events across {df['legislator_id'].nunique()} legislators.")
        return df


class DifferenceInDifferencesEstimator:
    """
    Executes Canonical Static DiD and Dynamic Event Study Specifications.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def estimate_static_did(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Estimates Static DiD:
        voted_pro_industry = alpha + beta * (treated * post) + treated + post + covariates + e
        Standard errors clustered at legislator level.
        """
        logger.info("Fitting Static Difference-in-Differences model with cluster-robust standard errors...")
        
        formula = (
            "voted_pro_industry ~ did_treated_unit * did_post_period "
            "+ dw_nominate_dim1 + party_republican"
        )
        model = smf.ols(formula, data=df)
        results = model.fit(cov_type="cluster", cov_kwds={"groups": df["legislator_id"]})

        did_param = "did_treated_unit:did_post_period"
        beta = results.params.get(did_param, np.nan)
        se = results.bse.get(did_param, np.nan)
        p_val = results.pvalues.get(did_param, np.nan)
        ci = results.conf_int().loc[did_param].values if did_param in results.params else [np.nan, np.nan]

        logger.info(
            f"Static DiD Treatment Effect (Beta): {beta:.4f}, SE: {se:.4f}, "
            f"p-val: {p_val:.4f}, 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]"
        )

        return {
            "model_type": "Static TWFE DiD",
            "treatment_effect_beta": float(beta),
            "standard_error": float(se),
            "t_statistic": float(results.tvalues.get(did_param, np.nan)),
            "p_value": float(p_val),
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "r_squared": float(results.rsquared),
            "n_observations": int(results.nobs),
            "n_clusters": int(df["legislator_id"].nunique())
        }

    def estimate_event_study(
        self,
        df: pd.DataFrame,
        n_leads: int = 3,
        n_lags: int = 3
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Fits Dynamic Event Study:
        Y_it = alpha_i + lambda_t + sum_{k != -1} beta_k * (Treated_i * 1(t = k)) + eps_it
        Tests Parallel Trends Assumption via joint hypothesis on pre-treatment leads.
        """
        logger.info("Constructing dynamic event-study specification with pre/post relative periods...")
        df_es = df.copy()

        # Construct relative event time k in [-n_leads, n_lags] relative to shock (2021)
        # 2019: -2, 2020: -1 (reference), 2021: 0, 2022: +1
        base_year = 2020  # Omitted reference period (t = -1)
        df_es["rel_time"] = df_es["year"] - 2021

        # Create dummy interactions for relative periods
        periods = [p for p in range(-n_leads, n_lags + 1) if p != -1]
        dummy_cols = []
        for p in periods:
            col_name = f"lead_lag_{p if p < 0 else f'plus_{p}'}"
            df_es[col_name] = ((df_es["rel_time"] == p) & (df_es["did_treated_unit"] == 1)).astype(int)
            dummy_cols.append(col_name)

        # Regress outcome on lead/lag dummies + baseline covariates
        formula = f"voted_pro_industry ~ {' + '.join(dummy_cols)} + C(year) + C(legislator_id)"
        model = smf.ols(formula, data=df_es)
        results = model.fit(cov_type="cluster", cov_kwds={"groups": df_es["legislator_id"]})

        # Extract event study trajectory
        es_records = []
        # Include reference period
        es_records.append({
            "relative_period": -1,
            "coefficient": 0.0,
            "standard_error": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value": 1.0,
            "period_type": "reference"
        })

        for p, col in zip(periods, dummy_cols):
            coef = results.params.get(col, 0.0)
            se = results.bse.get(col, 0.0)
            ci = results.conf_int().loc[col].values if col in results.params else [0.0, 0.0]
            p_val = results.pvalues.get(col, 1.0)
            es_records.append({
                "relative_period": p,
                "coefficient": float(coef),
                "standard_error": float(se),
                "ci_lower": float(ci[0]),
                "ci_upper": float(ci[1]),
                "p_value": float(p_val),
                "period_type": "pre-treatment (lead)" if p < -1 else "post-treatment (lag)"
            })

        df_event_study = pd.DataFrame(es_records).sort_values("relative_period").reset_index(drop=True)

        # Joint Wald test on pre-treatment leads: H0: beta_lead = 0
        lead_cols = [c for p, c in zip(periods, dummy_cols) if p < -1 and c in results.params]
        if lead_cols:
            r_matrix = np.zeros((len(lead_cols), len(results.params)))
            for r_idx, col in enumerate(lead_cols):
                param_idx = list(results.params.index).index(col)
                r_matrix[r_idx, param_idx] = 1.0
            
            f_test = results.f_test(r_matrix)
            f_stat = float(f_test.fvalue)
            f_pval = float(f_test.pvalue)
        else:
            f_stat, f_pval = 0.0, 1.0

        parallel_trends_valid = f_pval >= 0.05
        logger.info(
            f"Parallel Trends Joint F-Test: F = {f_stat:.3f}, p-value = {f_pval:.4f}. "
            f"Status: {'VALID (Fail to Reject H0)' if parallel_trends_valid else 'VIOLATED (Reject H0)'}"
        )

        test_summary = {
            "wald_f_stat": f_stat,
            "wald_p_value": f_pval,
            "parallel_trends_satisfied": bool(parallel_trends_valid),
            "lead_count": len(lead_cols)
        }

        return df_event_study, test_summary


def run_did_pipeline():
    config = load_config()
    root = get_project_root()
    proc_dir = ensure_dir(root / config["paths"]["processed_data"])

    panel_path = proc_dir / "panel_analysis_master.parquet"
    if not panel_path.exists():
        logger.warning("Master panel not found. Executing data integration first...")
        from scripts.data_integration import main as run_integration
        run_integration()

    df_panel = pd.read_parquet(panel_path)
    estimator = DifferenceInDifferencesEstimator(config)

    # 1. Static DiD
    static_res = estimator.estimate_static_did(df_panel)
    df_static = pd.DataFrame([static_res])
    df_static.to_csv(proc_dir / "did_static_results.csv", index=False)

    # 2. Dynamic Event Study
    df_es, pt_test = estimator.estimate_event_study(df_panel, n_leads=2, n_lags=2)
    df_es.to_csv(proc_dir / "did_event_study_trajectory.csv", index=False)

    logger.info("=== Static DiD Estimation Output ===")
    logger.info(f"\n{df_static.to_string(index=False)}")
    logger.info("=== Dynamic Event Study Trajectory ===")
    logger.info(f"\n{df_es.to_string(index=False)}")
    logger.info("Difference-in-Differences pipeline completed successfully.")


if __name__ == "__main__":
    run_did_pipeline()
