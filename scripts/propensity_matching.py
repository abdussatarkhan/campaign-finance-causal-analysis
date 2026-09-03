"""
Propensity Score Matching (PSM) Engine:
1. Logistic Regression Propensity Score Estimation on Ideological & Demographic Covariates
2. Nearest-Neighbor Matching with Caliper Restriction
3. Rigorous Balance Diagnostics: Standardized Mean Differences (SMD), Variance Ratios, Love Plot Data
4. Average Treatment Effect on the Treated (ATT) Estimation on Matched Sample
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import setup_logger, load_config, get_project_root, ensure_dir, compute_standardized_mean_difference

logger = setup_logger("propensity_matching")


class PropensityScoreMatcher:
    """
    Implements Propensity Score Matching to estimate the causal impact of receiving
    heavy industry campaign donations on subsequent roll-call voting behavior.
    """

    def __init__(
        self,
        treatment_col: str = "high_donation_treatment",
        outcome_col: str = "voted_pro_industry",
        covariates: Optional[List[str]] = None,
        caliper: float = 0.05,
        replace: bool = False
    ):
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.covariates = covariates or [
            "dw_nominate_dim1",
            "dw_nominate_dim2",
            "party_republican",
            "tenure_years",
            "median_district_income",
            "college_educated_pct",
            "urban_population_pct"
        ]
        self.caliper = caliper
        self.replace = replace
        self.ps_model: Optional[sm.Logit] = None
        self.ps_results: Optional[Any] = None

    def fit_propensity_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Estimates propensity scores via logistic regression.
        P(Treatment = 1 | X)
        """
        logger.info("Estimating propensity scores via multivariate logistic regression...")
        df_clean = df.dropna(subset=[self.treatment_col, self.outcome_col] + self.covariates).copy()

        X = sm.add_constant(df_clean[self.covariates])
        y = df_clean[self.treatment_col].astype(int)

        self.ps_model = sm.Logit(y, X)
        self.ps_results = self.ps_model.fit(disp=False)
        logger.info(f"Logit Pseudo R-squared: {self.ps_results.prsquared:.4f}")

        # Predict propensity scores and logit of propensity scores
        df_clean["propensity_score"] = self.ps_results.predict(X)
        # Avoid infinity in logit transform
        eps = 1e-6
        clipped_ps = np.clip(df_clean["propensity_score"], eps, 1 - eps)
        df_clean["ps_logit"] = np.log(clipped_ps / (1.0 - clipped_ps))

        return df_clean

    def match_nearest_neighbors(self, df_scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Performs 1:1 Nearest-Neighbor matching on logit propensity scores with caliper.
        Returns:
            matched_df: DataFrame containing matched pairs
            balance_table: DataFrame summarizing balance statistics pre/post match
        """
        logger.info(f"Conducting nearest-neighbor matching (caliper={self.caliper}, replace={self.replace})...")
        treated = df_scored[df_scored[self.treatment_col] == 1].copy()
        control = df_scored[df_scored[self.treatment_col] == 0].copy()

        caliper_dist = self.caliper * df_scored["ps_logit"].std()
        logger.info(f"Absolute logit caliper distance: {caliper_dist:.4f}")

        matched_treated_indices = []
        matched_control_indices = []
        available_controls = control.index.tolist()

        # Iterate over treated units
        for t_idx, t_row in treated.iterrows():
            if not available_controls:
                break

            ctrl_subset = control.loc[available_controls]
            distances = np.abs(ctrl_subset["ps_logit"] - t_row["ps_logit"])
            min_dist = distances.min()

            if min_dist <= caliper_dist:
                best_match_idx = distances.idxmin()
                matched_treated_indices.append(t_idx)
                matched_control_indices.append(best_match_idx)

                if not self.replace:
                    available_controls.remove(best_match_idx)

        logger.info(f"Matched {len(matched_treated_indices)} out of {len(treated)} treated units.")

        matched_treated_df = df_scored.loc[matched_treated_indices].copy()
        matched_treated_df["matched_pair_id"] = range(len(matched_treated_indices))
        matched_control_df = df_scored.loc[matched_control_indices].copy()
        matched_control_df["matched_pair_id"] = range(len(matched_control_indices))

        matched_df = pd.concat([matched_treated_df, matched_control_df], ignore_index=True)

        # Compute Balance Diagnostics
        balance_table = self.compute_balance_diagnostics(df_scored, matched_df)
        return matched_df, balance_table

    def compute_balance_diagnostics(self, df_unmatched: pd.DataFrame, df_matched: pd.DataFrame) -> pd.DataFrame:
        """
        Computes Cohen's SMD and Variance Ratios before and after matching for each covariate.
        Target threshold: |SMD| < 0.10 indicates strong balance.
        """
        logger.info("Evaluating covariate balance diagnostics...")
        records = []

        unmatched_t = df_unmatched[df_unmatched[self.treatment_col] == 1]
        unmatched_c = df_unmatched[df_unmatched[self.treatment_col] == 0]

        matched_t = df_matched[df_matched[self.treatment_col] == 1]
        matched_c = df_matched[df_matched[self.treatment_col] == 0]

        for cov in self.covariates:
            # Unmatched
            smd_pre = compute_standardized_mean_difference(unmatched_t[cov].values, unmatched_c[cov].values)
            var_ratio_pre = unmatched_t[cov].var() / (unmatched_c[cov].var() + 1e-12)

            # Matched
            smd_post = compute_standardized_mean_difference(matched_t[cov].values, matched_c[cov].values)
            var_ratio_post = matched_t[cov].var() / (matched_c[cov].var() + 1e-12)

            t_stat, p_val = stats.ttest_ind(matched_t[cov].dropna(), matched_c[cov].dropna(), equal_var=False)

            records.append({
                "covariate": cov,
                "smd_pre_match": round(smd_pre, 4),
                "smd_post_match": round(smd_post, 4),
                "smd_reduction_pct": round((1.0 - abs(smd_post) / (abs(smd_pre) + 1e-12)) * 100, 2),
                "var_ratio_pre": round(var_ratio_pre, 3),
                "var_ratio_post": round(var_ratio_post, 3),
                "post_match_pvalue": round(p_val, 4),
                "balanced_status": "BALANCED" if abs(smd_post) <= 0.10 else "IMBALANCED"
            })

        balance_df = pd.DataFrame(records)
        return balance_df

    def estimate_att(self, df_matched: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes Average Treatment Effect on the Treated (ATT) on matched sample.
        Outcome: voted_pro_industry.
        """
        matched_t = df_matched[df_matched[self.treatment_col] == 1][self.outcome_col]
        matched_c = df_matched[df_matched[self.treatment_col] == 0][self.outcome_col]

        y_t = matched_t.mean()
        y_c = matched_c.mean()
        att = y_t - y_c

        # Paired t-test
        t_stat, p_val = stats.ttest_rel(matched_t.values, matched_c.values)
        diff = matched_t.values - matched_c.values
        se = np.std(diff, ddof=1) / np.sqrt(len(diff))
        ci_lower = att - 1.96 * se
        ci_upper = att + 1.96 * se

        results = {
            "att": float(att),
            "treated_outcome_mean": float(y_t),
            "control_outcome_mean": float(y_c),
            "standard_error": float(se),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "sample_size_matched_pairs": len(matched_t)
        }
        logger.info(f"ATT Estimate: {att:.4f} (SE: {se:.4f}, p-value: {p_val:.4e}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
        return results


def run_matching_pipeline():
    config = load_config()
    root = get_project_root()
    proc_dir = ensure_dir(root / config["paths"]["processed_data"])
    models_dir = ensure_dir(root / config["paths"]["models_dir"])

    panel_path = proc_dir / "panel_analysis_master.parquet"
    if not panel_path.exists():
        logger.warning("Panel data not found. Running data integration first...")
        from scripts.data_integration import main as run_integration
        run_integration()

    df_panel = pd.read_parquet(panel_path)

    matcher = PropensityScoreMatcher(
        treatment_col="high_donation_treatment",
        outcome_col="voted_pro_industry",
        caliper=config["causal_inference"]["propensity_score_matching"]["caliper"],
        replace=config["causal_inference"]["propensity_score_matching"]["replace"]
    )

    df_scored = matcher.fit_propensity_scores(df_panel)
    df_matched, balance_table = matcher.match_nearest_neighbors(df_scored)
    att_results = matcher.estimate_att(df_matched)

    # Save matched artifacts
    matched_out = proc_dir / "psm_matched_dataset.parquet"
    df_matched.to_parquet(matched_out, index=False)

    balance_out = proc_dir / "psm_balance_diagnostics.csv"
    balance_table.to_csv(balance_out, index=False)

    logger.info("=== Covariate Balance Summary ===")
    logger.info(f"\n{balance_table.to_string(index=False)}")
    logger.info("=== Treatment Effect Summary (ATT) ===")
    for k, v in att_results.items():
        logger.info(f"  {k}: {v}")

    logger.info("Propensity score matching pipeline executed successfully.")


if __name__ == "__main__":
    run_matching_pipeline()
