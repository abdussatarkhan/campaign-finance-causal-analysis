"""
Hypothesis Testing and Clustered Logistic Regression Engine:
1. Multi-Specification Logistic Regression with Standard Errors Clustered by Legislator
2. Stepwise Control Architecture: Unadjusted -> Partisan -> Ideological (DW-NOMINATE) -> Fully Conditioned
3. Odds Ratio (OR) and Average Marginal Effects (AME) Calculation
4. Rigorous Statistical Testing of the Null Hypothesis of Zero Campaign Contribution Influence
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import setup_logger, load_config, get_project_root, ensure_dir

logger = setup_logger("hypothesis_testing")


class ClusteredLogisticEstimator:
    """
    Fits logistic regression models with cluster-robust standard errors
    to evaluate how financial backing affects the probability of casting a pro-industry vote.
    """

    def __init__(self, cluster_col: str = "legislator_id"):
        self.cluster_col = cluster_col

    def fit_model_specification(
        self,
        df: pd.DataFrame,
        formula: str,
        spec_name: str
    ) -> Dict[str, Any]:
        """
        Fits Logit model using statsmodels GLM/Logit with clustered sandwich covariance.
        """
        logger.info(f"Estimating {spec_name}: {formula}")
        df_valid = df.dropna(subset=[col.strip() for col in formula.replace("~", "+").replace("*", "+").replace("C(", "").replace(")", "").split("+") if col.strip() in df.columns]).copy()

        model = smf.logit(formula, data=df_valid)
        fit_res = model.fit(
            cov_type="cluster",
            cov_kwds={"groups": df_valid[self.cluster_col]},
            disp=False
        )

        # Focus on the donation parameter: log_donations
        target_param = "log_donations"
        if target_param in fit_res.params:
            beta = fit_res.params[target_param]
            se = fit_res.bse[target_param]
            z_stat = fit_res.tvalues[target_param]
            p_val = fit_res.pvalues[target_param]
            odds_ratio = np.exp(beta)
            ci = fit_res.conf_int().loc[target_param].values
            or_ci = np.exp(ci)
        else:
            beta, se, z_stat, p_val, odds_ratio = np.nan, np.nan, np.nan, np.nan, np.nan
            ci, or_ci = [np.nan, np.nan], [np.nan, np.nan]

        # Compute Average Marginal Effect (AME) for log_donations
        # AME = mean( p * (1 - p) * beta )
        preds = fit_res.predict()
        derivative_factor = preds * (1.0 - preds)
        ame = float(np.mean(derivative_factor) * beta)
        ame_se = float(np.mean(derivative_factor) * se)

        # Star annotation
        stars = ""
        if p_val < 0.001:
            stars = "***"
        elif p_val < 0.01:
            stars = "**"
        elif p_val < 0.05:
            stars = "*"

        return {
            "Specification": spec_name,
            "Formula": formula,
            "Donations_Beta": float(beta),
            "Donations_SE": float(se),
            "Odds_Ratio": float(odds_ratio),
            "OR_95_CI_Lower": float(or_ci[0]),
            "OR_95_CI_Upper": float(or_ci[1]),
            "AME": float(ame),
            "AME_SE": float(ame_se),
            "Z_Statistic": float(z_stat),
            "P_Value": float(p_val),
            "Significance": stars,
            "Pseudo_R2": float(fit_res.prsquared),
            "Log_Likelihood": float(fit_res.llf),
            "N_Obs": int(fit_res.nobs),
            "N_Clusters": int(df_valid[self.cluster_col].nunique())
        }

    def run_multi_specification_suite(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs progressive econometric specifications to observe parameter stability.
        """
        specifications = [
            (
                "Model 1: Bivariate Baseline",
                "voted_pro_industry ~ log_donations"
            ),
            (
                "Model 2: Partisan Control",
                "voted_pro_industry ~ log_donations + party_republican"
            ),
            (
                "Model 3: Ideological Coordinates",
                "voted_pro_industry ~ log_donations + party_republican + dw_nominate_dim1 + dw_nominate_dim2"
            ),
            (
                "Model 4: Fully Conditioned Econometric Model",
                "voted_pro_industry ~ log_donations + party_republican + dw_nominate_dim1 + dw_nominate_dim2 + tenure_years + median_district_income + college_educated_pct + urban_population_pct + C(industry_code)"
            )
        ]

        results = []
        for name, formula in specifications:
            res = self.fit_model_specification(df, formula, name)
            results.append(res)

        df_results = pd.DataFrame(results)
        return df_results


def run_hypothesis_pipeline():
    config = load_config()
    root = get_project_root()
    proc_dir = ensure_dir(root / config["paths"]["processed_data"])

    panel_path = proc_dir / "panel_analysis_master.parquet"
    if not panel_path.exists():
        logger.warning("Master panel not found. Executing data integration first...")
        from scripts.data_integration import main as run_integration
        run_integration()

    df_panel = pd.read_parquet(panel_path)
    estimator = ClusteredLogisticEstimator(cluster_col="legislator_id")
    df_results = estimator.run_multi_specification_suite(df_panel)

    out_file = proc_dir / "hypothesis_testing_specifications.csv"
    df_results.to_csv(out_file, index=False)

    logger.info("=== Clustered Logistic Regression Model Comparison ===")
    display_cols = ["Specification", "Donations_Beta", "Donations_SE", "Odds_Ratio", "AME", "P_Value", "Significance", "Pseudo_R2", "N_Obs"]
    logger.info(f"\n{df_results[display_cols].to_string(index=False)}")
    logger.info("Hypothesis testing completed successfully.")


if __name__ == "__main__":
    run_hypothesis_pipeline()
