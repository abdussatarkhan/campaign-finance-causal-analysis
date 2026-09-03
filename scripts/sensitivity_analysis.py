"""
Sensitivity and Robustness Analysis Engine:
1. RDD Bandwidth Perturbation & Alternative Kernels
2. RDD Placebo Cutoff Falsification Tests
3. Propensity Score Matching Caliper and Replacement Sensitivity
4. Subgroup Industry Heterogeneity & Leave-One-Industry-Out Checks
"""

import sys
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import setup_logger, load_config, get_project_root, ensure_dir
from scripts.regression_discontinuity import LocalPolynomialRDD, ImbensKalyanaramanBandwidth
from scripts.propensity_matching import PropensityScoreMatcher

logger = setup_logger("sensitivity_analysis")


class SensitivitySuite:
    """Executes systematic econometric sensitivity checks across models."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def rdd_bandwidth_sensitivity(self, running_var: np.ndarray, outcome_var: np.ndarray) -> pd.DataFrame:
        """
        Tests stability of RDD treatment effect across multiple bandwidth multipliers (0.5h to 2.0h).
        """
        logger.info("Executing RDD Bandwidth Sensitivity Sweep...")
        h_base = ImbensKalyanaramanBandwidth.calculate_bandwidth(running_var, outcome_var, cutoff=0.0)
        multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
        results = []

        for m in multipliers:
            h_test = h_base * m
            rdd = LocalPolynomialRDD(cutoff=0.0, polynomial_degree=1, kernel="triangular", bandwidth=h_test)
            try:
                res = rdd.fit(running_var, outcome_var)
                results.append({
                    "Bandwidth_Multiplier": m,
                    "Bandwidth_Value": round(h_test, 4),
                    "Kernel": "triangular",
                    "Tau_Estimate": round(res["treatment_effect_tau"], 4),
                    "Standard_Error": round(res["standard_error"], 4),
                    "P_Value": round(res["p_value"], 4),
                    "N_Effective": res["effective_sample_size"]
                })
            except Exception as e:
                logger.warning(f"Failed RDD at bandwidth {h_test:.4f}: {e}")

        return pd.DataFrame(results)

    def rdd_placebo_cutoffs(self, running_var: np.ndarray, outcome_var: np.ndarray) -> pd.DataFrame:
        """
        Falsification test: Re-estimates RDD at artificial, false cutoffs where no jump should occur.
        """
        logger.info("Executing RDD Placebo Cutoff Falsification Tests...")
        placebo_cutoffs = [-0.10, -0.05, 0.05, 0.10]
        results = []

        for c_fake in placebo_cutoffs:
            rdd = LocalPolynomialRDD(cutoff=c_fake, polynomial_degree=1, kernel="triangular", bandwidth=0.06)
            try:
                res = rdd.fit(running_var, outcome_var)
                results.append({
                    "Cutoff_Type": f"Placebo ({c_fake:+.2f})",
                    "Cutoff_Value": c_fake,
                    "Tau_Estimate": round(res["treatment_effect_tau"], 4),
                    "Standard_Error": round(res["standard_error"], 4),
                    "P_Value": round(res["p_value"], 4),
                    "Null_Hypothesis_Held": res["p_value"] >= 0.05
                })
            except Exception as e:
                logger.warning(f"Failed placebo at cutoff {c_fake}: {e}")

        return pd.DataFrame(results)

    def psm_specification_sensitivity(self, df_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Tests sensitivity of PSM ATT across varying caliper tolerances and replacement schemes.
        """
        logger.info("Evaluating PSM Caliper and Replacement Sensitivity...")
        calipers = [0.01, 0.02, 0.05, 0.10]
        results = []

        for replace in [False, True]:
            for cal in calipers:
                matcher = PropensityScoreMatcher(
                    treatment_col="high_donation_treatment",
                    outcome_col="voted_pro_industry",
                    caliper=cal,
                    replace=replace
                )
                try:
                    df_scored = matcher.fit_propensity_scores(df_panel)
                    df_matched, balance_df = matcher.match_nearest_neighbors(df_scored)
                    att_dict = matcher.estimate_att(df_matched)

                    # Maximum remaining standardized mean difference across covariates
                    max_abs_smd = balance_df["smd_post_match"].abs().max()

                    results.append({
                        "Replacement": replace,
                        "Caliper": cal,
                        "Matched_Pairs": att_dict["sample_size_matched_pairs"],
                        "ATT": round(att_dict["att"], 4),
                        "Standard_Error": round(att_dict["standard_error"], 4),
                        "P_Value": round(att_dict["p_value"], 4),
                        "Max_Abs_SMD": round(max_abs_smd, 4),
                        "Balance_Satisfied": max_abs_smd <= 0.10
                    })
                except Exception as e:
                    logger.warning(f"PSM sensitivity failed for Caliper={cal}, Replace={replace}: {e}")

        return pd.DataFrame(results)

    def industry_heterogeneity(self, df_panel: pd.DataFrame) -> pd.DataFrame:
        """
        Estimates treatment effect independently within each target economic sector.
        """
        logger.info("Estimating Industry-Specific Subgroup Treatment Heterogeneity...")
        industries = df_panel["industry_code"].unique()
        results = []

        for ind in industries:
            sub_df = df_panel[df_panel["industry_code"] == ind]
            if len(sub_df) < 50:
                continue

            matcher = PropensityScoreMatcher(
                treatment_col="high_donation_treatment",
                outcome_col="voted_pro_industry",
                caliper=0.08,
                replace=False
            )
            try:
                scored = matcher.fit_propensity_scores(sub_df)
                matched, _ = matcher.match_nearest_neighbors(scored)
                att_res = matcher.estimate_att(matched)

                results.append({
                    "Industry_Code": ind,
                    "Total_Votes": len(sub_df),
                    "Matched_Pairs": att_res["sample_size_matched_pairs"],
                    "Subgroup_ATT": round(att_res["att"], 4),
                    "Standard_Error": round(att_res["standard_error"], 4),
                    "P_Value": round(att_res["p_value"], 4)
                })
            except Exception as e:
                logger.warning(f"Subgroup analysis failed for industry {ind}: {e}")

        return pd.DataFrame(results)


def run_sensitivity_pipeline():
    config = load_config()
    root = get_project_root()
    proc_dir = ensure_dir(root / config["paths"]["processed_data"])

    panel_path = proc_dir / "panel_analysis_master.parquet"
    if not panel_path.exists():
        logger.warning("Master panel missing. Executing data integration first...")
        from scripts.data_integration import main as run_integration
        run_integration()

    df_panel = pd.read_parquet(panel_path)
    suite = SensitivitySuite(config)

    # 1. RDD Bandwidth Sweep
    leg_rdd = df_panel.groupby("legislator_id").first().reset_index()
    df_bw = suite.rdd_bandwidth_sensitivity(leg_rdd["vote_margin"].values, leg_rdd["log_donations"].values)
    df_bw.to_csv(proc_dir / "sensitivity_rdd_bandwidth.csv", index=False)

    # 2. RDD Placebo Cutoffs
    df_placebo = suite.rdd_placebo_cutoffs(leg_rdd["vote_margin"].values, leg_rdd["log_donations"].values)
    df_placebo.to_csv(proc_dir / "sensitivity_rdd_placebo.csv", index=False)

    # 3. PSM Matching Sensitivity
    df_psm = suite.psm_specification_sensitivity(df_panel)
    df_psm.to_csv(proc_dir / "sensitivity_psm_calipers.csv", index=False)

    # 4. Industry Subgroup Heterogeneity
    df_ind = suite.industry_heterogeneity(df_panel)
    df_ind.to_csv(proc_dir / "sensitivity_industry_subgroups.csv", index=False)

    logger.info("=== Sensitivity Analysis Summary Completed ===")
    logger.info(f"RDD Bandwidth Sweep:\n{df_bw.to_string(index=False)}")
    logger.info(f"RDD Placebo Falsification:\n{df_placebo.to_string(index=False)}")
    logger.info(f"PSM Caliper Stability:\n{df_psm.to_string(index=False)}")
    logger.info(f"Industry Heterogeneity:\n{df_ind.to_string(index=False)}")


if __name__ == "__main__":
    run_sensitivity_pipeline()
