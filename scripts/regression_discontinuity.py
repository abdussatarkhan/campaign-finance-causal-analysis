"""
Regression Discontinuity Design (RDD) Causal Analysis:
1. Imbens-Kalyanaraman (IK) Optimal Bandwidth Selection for Sharp RDD
2. Local Polynomial Regression (Local Linear and Quadratic with Triangular/Epanechnikov Kernels)
3. McCrary (2008) Density Discontinuity Test for Running Variable Manipulation
4. Close-Election Treatment Effect Estimation on Campaign Finance Receipts and Voting Behavior
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import (
    setup_logger, load_config, get_project_root, ensure_dir,
    triangular_kernel, epanechnikov_kernel, uniform_kernel
)

logger = setup_logger("regression_discontinuity")


class ImbensKalyanaramanBandwidth:
    """
    Implements the Imbens-Kalyanaraman (2012) data-driven optimal bandwidth
    selection algorithm for sharp Regression Discontinuity Designs.
    """

    @classmethod
    def calculate_bandwidth(
        cls,
        running_var: np.ndarray,
        outcome_var: np.ndarray,
        cutoff: float = 0.0,
        kernel: str = "triangular"
    ) -> float:
        """
        Calculates optimal bandwidth h_IK around cutoff c.
        """
        x = running_var - cutoff
        y = outcome_var
        n = len(x)

        # Step 1: Preliminary bandwidth h1 using Silverman's rule of thumb
        s_x = np.std(x, ddof=1)
        h1 = 1.84 * s_x * (n ** (-0.2))

        # Filter within pilot bandwidth
        mask_pilot = np.abs(x) <= h1
        x_pilot, y_pilot = x[mask_pilot], y[mask_pilot]

        if len(x_pilot) < 30:
            return float(max(0.04, s_x * 0.5))

        # Estimate density f(c)
        n_left = np.sum((x < 0) & mask_pilot)
        n_right = np.sum((x >= 0) & mask_pilot)
        f_hat_c = (n_left + n_right) / (2.0 * n * h1)

        # Step 2: Estimate variance sigma^2(c)
        treated_pilot = x_pilot >= 0
        var_left = np.var(y_pilot[~treated_pilot], ddof=1) if np.sum(~treated_pilot) > 5 else np.var(y, ddof=1)
        var_right = np.var(y_pilot[treated_pilot], ddof=1) if np.sum(treated_pilot) > 5 else np.var(y, ddof=1)
        sigma2_hat = (var_left + var_right) / 2.0

        # Step 3: Estimate second derivative m''(c) using 3rd order polynomial
        X_poly = np.column_stack([np.ones_like(x_pilot), x_pilot, x_pilot**2, x_pilot**3, x_pilot >= 0])
        try:
            poly_model = sm.OLS(y_pilot, X_poly).fit()
            m2_diff = 2.0 * poly_model.params[2]
            # Regularization term to prevent denominator explosion
            m2_term = max(m2_diff**2, 0.01)
        except Exception:
            m2_term = 1.0

        # Kernel constant: C_k = 3.4375 for triangular, 5.4 for uniform
        c_k = 3.4375 if kernel == "triangular" else 5.400
        
        # IK optimal bandwidth formula
        h_opt = c_k * ((sigma2_hat / (f_hat_c * m2_term + 1e-6)) ** 0.2) * (n ** (-0.2))
        h_opt = float(np.clip(h_opt, 0.02, 0.20))
        logger.info(f"Imbens-Kalyanaraman computed optimal bandwidth: h = {h_opt:.4f}")
        return h_opt


class McCraryDensityTest:
    """
    Implements the McCrary (2008) density test to detect precise sorting or manipulation
    of the running variable around the election cutoff.
    """

    @staticmethod
    def run_test(running_var: np.ndarray, cutoff: float = 0.0, n_bins: int = 40) -> Dict[str, Any]:
        """
        Tests null hypothesis of continuity of running variable density at cutoff.
        """
        logger.info("Executing McCrary density continuity test around cutoff = 0.0...")
        x = running_var[~np.isnan(running_var)]
        
        # Bin frequencies
        left_mask = x < cutoff
        right_mask = x >= cutoff

        bins_left = np.linspace(cutoff - 0.25, cutoff, n_bins // 2 + 1)
        bins_right = np.linspace(cutoff, cutoff + 0.25, n_bins // 2 + 1)

        counts_left, edges_left = np.histogram(x[left_mask], bins=bins_left)
        counts_right, edges_right = np.histogram(x[right_mask], bins=bins_right)

        mids_left = (edges_left[:-1] + edges_left[1:]) / 2.0
        mids_right = (edges_right[:-1] + edges_right[1:]) / 2.0

        # Fit local regressions on bin counts
        X_l = sm.add_constant(mids_left - cutoff)
        X_r = sm.add_constant(mids_right - cutoff)

        model_l = sm.OLS(counts_left, X_l).fit()
        model_r = sm.OLS(counts_right, X_r).fit()

        f_left = max(model_l.params[0], 1.0)
        f_right = max(model_r.params[0], 1.0)

        theta = np.log(f_right) - np.log(f_left)
        se_theta = np.sqrt(1.0 / np.sum(counts_left) + 1.0 / np.sum(counts_right))
        z_score = theta / se_theta
        p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z_score)))

        manipulation_detected = p_value < 0.05
        logger.info(f"McCrary Log Difference (theta): {theta:.4f}, SE: {se_theta:.4f}, z: {z_score:.3f}, p-value: {p_value:.4f}")
        logger.info(f"Evidence of manipulation at threshold: {'YES (Reject Null)' if manipulation_detected else 'NO (Fail to Reject)'}")

        return {
            "theta_log_diff": float(theta),
            "se_theta": float(se_theta),
            "z_score": float(z_score),
            "p_value": float(p_value),
            "manipulation_detected": bool(manipulation_detected)
        }


class LocalPolynomialRDD:
    """
    Estimates Treatment Effects using Local Polynomial Non-Parametric WLS Regression.
    """

    def __init__(
        self,
        cutoff: float = 0.0,
        polynomial_degree: int = 1,
        kernel: str = "triangular",
        bandwidth: Optional[float] = None
    ):
        self.cutoff = cutoff
        self.degree = polynomial_degree
        self.kernel_name = kernel
        self.bandwidth = bandwidth
        self.model_results: Optional[Any] = None

    def _get_kernel_weights(self, u: np.ndarray) -> np.ndarray:
        if self.kernel_name == "triangular":
            return triangular_kernel(u)
        elif self.kernel_name == "epanechnikov":
            return epanechnikov_kernel(u)
        else:
            return uniform_kernel(u)

    def fit(self, running_var: np.ndarray, outcome_var: np.ndarray) -> Dict[str, Any]:
        """
        Fits WLS regression within bandwidth window:
        Y = a + tau * D + b1 * X + b2 * D * X + [higher poly terms]
        """
        valid_idx = ~(np.isnan(running_var) | np.isnan(outcome_var))
        x_raw = running_var[valid_idx]
        y = outcome_var[valid_idx]

        # Determine bandwidth if not specified
        if self.bandwidth is None:
            self.bandwidth = ImbensKalyanaramanBandwidth.calculate_bandwidth(
                x_raw, y, cutoff=self.cutoff, kernel=self.kernel_name
            )

        # Center running variable at cutoff
        x = x_raw - self.cutoff
        in_window = np.abs(x) <= self.bandwidth
        
        x_win = x[in_window]
        y_win = y[in_window]
        n_obs = len(x_win)

        if n_obs < 15:
            raise ValueError(f"Insufficient observations ({n_obs}) within bandwidth h={self.bandwidth}")

        # Treatment indicator: D = 1(X >= 0)
        d_win = (x_win >= 0).astype(float)
        u = x_win / self.bandwidth
        weights = self._get_kernel_weights(u)

        # Construct Design Matrix
        features = [np.ones_like(x_win), d_win, x_win, d_win * x_win]
        feature_names = ["Intercept", "Treatment_RD", "Running_Var", "Interaction"]

        if self.degree == 2:
            features.extend([x_win**2, d_win * (x_win**2)])
            feature_names.extend(["Running_Var_Sq", "Interaction_Sq"])

        X_mat = np.column_stack(features)

        # Weighted Least Squares (WLS)
        wls_model = sm.WLS(y_win, X_mat, weights=weights)
        self.model_results = wls_model.fit(cov_type="HC1")

        tau_idx = 1
        tau = float(self.model_results.params[tau_idx])
        se = float(self.model_results.bse[tau_idx])
        t_stat = float(self.model_results.tvalues[tau_idx])
        p_val = float(self.model_results.pvalues[tau_idx])
        ci = self.model_results.conf_int()[tau_idx]

        logger.info(
            f"RDD Local Poly (Deg={self.degree}, Kernel={self.kernel_name}, h={self.bandwidth:.4f}): "
            f"Tau = {tau:.4f}, SE = {se:.4f}, p = {p_val:.4f}, 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}], N={n_obs}"
        )

        return {
            "treatment_effect_tau": tau,
            "standard_error": se,
            "t_statistic": t_stat,
            "p_value": p_val,
            "ci_lower": float(ci[0]),
            "ci_upper": float(ci[1]),
            "bandwidth": float(self.bandwidth),
            "effective_sample_size": int(n_obs),
            "poly_degree": int(self.degree),
            "kernel": self.kernel_name
        }


def run_rdd_pipeline():
    config = load_config()
    root = get_project_root()
    proc_dir = ensure_dir(root / config["paths"]["processed_data"])

    panel_path = proc_dir / "panel_analysis_master.parquet"
    if not panel_path.exists():
        logger.warning("Panel data missing. Running data integration first...")
        from scripts.data_integration import main as run_integration
        run_integration()

    df_panel = pd.read_parquet(panel_path)
    # Unique legislator-level observations for close election RDD
    leg_rdd = df_panel.groupby("legislator_id").first().reset_index()

    # 1. McCrary Density Test on vote margin
    density_test = McCraryDensityTest.run_test(leg_rdd["vote_margin"].values, cutoff=0.0)

    # 2. Local Linear RDD on Industry Contributions
    rdd_linear = LocalPolynomialRDD(cutoff=0.0, polynomial_degree=1, kernel="triangular")
    res_linear = rdd_linear.fit(leg_rdd["vote_margin"].values, leg_rdd["log_donations"].values)

    # 3. Local Quadratic RDD on Industry Contributions
    rdd_quad = LocalPolynomialRDD(cutoff=0.0, polynomial_degree=2, kernel="triangular", bandwidth=res_linear["bandwidth"])
    res_quad = rdd_quad.fit(leg_rdd["vote_margin"].values, leg_rdd["log_donations"].values)

    # 4. RDD directly on Voting Outcome (Roll-Call level)
    rdd_vote = LocalPolynomialRDD(cutoff=0.0, polynomial_degree=1, kernel="triangular")
    res_vote = rdd_vote.fit(df_panel["vote_margin"].values, df_panel["voted_pro_industry"].values)

    rdd_summary = pd.DataFrame([
        {"Specification": "Local Linear (Donations)", **res_linear},
        {"Specification": "Local Quadratic (Donations)", **res_quad},
        {"Specification": "Local Linear (Pro-Industry Vote)", **res_vote}
    ])

    out_file = proc_dir / "rdd_estimation_results.csv"
    rdd_summary.to_csv(out_file, index=False)
    logger.info(f"RDD results saved to {out_file}:\n{rdd_summary.to_string(index=False)}")


if __name__ == "__main__":
    run_rdd_pipeline()
