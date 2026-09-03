"""
Utility functions, logging configuration, metric calculators, and synthetic
benchmark generators for the Campaign Finance Causal Analysis project.
"""

import os
import sys
import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
import pandas as pd


def get_project_root() -> Path:
    """Returns the absolute root directory of the repository."""
    current_path = Path(__file__).resolve()
    # Go up from scripts/ to project root
    return current_path.parent.parent


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Loads YAML configuration file from config/config.yaml or given path.
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_logger(name: str = "causal_logger", log_level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a standardized console and file logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """Ensures that a specified directory exists; creates it if absent."""
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ==============================================================================
# Econometric & Statistical Kernels
# ==============================================================================

def triangular_kernel(u: np.ndarray) -> np.ndarray:
    """
    Triangular kernel function K(u) = (1 - |u|) * 1(|u| <= 1).
    Standard optimal kernel for boundary regression discontinuity design.
    """
    abs_u = np.abs(u)
    return np.where(abs_u <= 1.0, 1.0 - abs_u, 0.0)


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    """
    Epanechnikov kernel function K(u) = 0.75 * (1 - u^2) * 1(|u| <= 1).
    Minimizes mean squared error in non-parametric estimation.
    """
    abs_u = np.abs(u)
    return np.where(abs_u <= 1.0, 0.75 * (1.0 - u**2), 0.0)


def uniform_kernel(u: np.ndarray) -> np.ndarray:
    """
    Uniform (rectangular) kernel function K(u) = 0.5 * 1(|u| <= 1).
    """
    return np.where(np.abs(u) <= 1.0, 0.5, 0.0)


def compute_standardized_mean_difference(
    treated: np.ndarray,
    control: np.ndarray,
    weights_treated: Optional[np.ndarray] = None,
    weights_control: Optional[np.ndarray] = None
) -> float:
    """
    Computes Cohen's Standardized Mean Difference (SMD) for balance assessment:
    SMD = (mean_t - mean_c) / sqrt((var_t + var_c) / 2)
    """
    if weights_treated is None:
        mean_t = np.nanmean(treated)
        var_t = np.nanvar(treated, ddof=1)
    else:
        norm_wt = weights_treated / np.sum(weights_treated)
        mean_t = np.sum(norm_wt * treated)
        var_t = np.sum(norm_wt * (treated - mean_t)**2)

    if weights_control is None:
        mean_c = np.nanmean(control)
        var_c = np.nanvar(control, ddof=1)
    else:
        norm_wc = weights_control / np.sum(weights_control)
        mean_c = np.sum(norm_wc * control)
        var_c = np.sum(norm_wc * (control - mean_c)**2)

    pooled_sd = np.sqrt(np.maximum((var_t + var_c) / 2.0, 1e-12))
    smd = (mean_t - mean_c) / pooled_sd
    return float(smd)


# ==============================================================================
# Synthetic Benchmark Dataset Generator (for End-to-End Execution)
# ==============================================================================

def generate_synthetic_election_and_voting_data(
    n_legislators: int = 435,
    n_bills_per_industry: int = 25,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generates realistic, econometrically grounded synthetic data for:
    1. Legislator master dataset (DW-NOMINATE scores, demographics, vote margins)
    2. FEC campaign contributions by industry PACs
    3. Roll-call voting records on industry-specific legislation
    """
    np.random.seed(random_seed)

    # 1. Legislators
    states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]
    parties = np.random.choice(["D", "R"], size=n_legislators, p=[0.51, 0.49])
    
    legislators = []
    for idx in range(n_legislators):
        party = parties[idx]
        leg_id = f"LEG_{1000 + idx}"
        bioguide_id = f"{'D' if party == 'D' else 'R'}{idx:04d}"
        fec_cand_id = f"H{idx:02d}{states[idx % len(states)]}{party}00"
        
        # DW-NOMINATE Dim 1: Liberal-Conservative spectrum (-1.0 to +1.0)
        dw_dim1 = np.random.normal(-0.45, 0.18) if party == "D" else np.random.normal(0.48, 0.17)
        dw_dim2 = np.random.normal(0.0, 0.22)
        
        # Vote margin in previous general election (-0.30 to +0.30 around cutoff 0)
        # Bimodal close margin for RDD analysis
        vote_margin = np.random.uniform(-0.15, 0.15) if np.random.rand() < 0.4 else np.random.uniform(-0.40, 0.40)
        
        tenure = np.random.choice(range(1, 15), p=np.exp(-0.15 * np.arange(14)) / np.sum(np.exp(-0.15 * np.arange(14))))
        median_income = np.random.normal(68000, 14000)
        college_pct = np.clip(np.random.normal(32, 8), 12, 65)
        urban_pct = np.clip(np.random.normal(65, 18), 15, 98)
        
        legislators.append({
            "legislator_id": leg_id,
            "bioguide_id": bioguide_id,
            "fec_cand_id": fec_cand_id,
            "name": f"Representative_{idx+1}",
            "party": party,
            "party_republican": 1 if party == "R" else 0,
            "state": states[idx % len(states)],
            "district": (idx % 25) + 1,
            "dw_nominate_dim1": dw_dim1,
            "dw_nominate_dim2": dw_dim2,
            "vote_margin": vote_margin,
            "tenure_years": tenure,
            "median_district_income": median_income,
            "college_educated_pct": college_pct,
            "urban_population_pct": urban_pct
        })
    df_legislators = pd.DataFrame(legislators)

    # 2. Campaign Contributions by Industry PAC
    industries = [
        {"code": "E01", "name": "Oil & Gas", "repub_lean": 0.35},
        {"code": "F07", "name": "Commercial Banks", "repub_lean": 0.15},
        {"code": "H01", "name": "Pharmaceuticals", "repub_lean": 0.10},
        {"code": "D01", "name": "Defense Aerospace", "repub_lean": 0.20}
    ]
    
    contributions = []
    cycles = [2018, 2020, 2022]
    quarters = ["Q1", "Q2", "Q3", "Q4"]
    
    for _, leg in df_legislators.iterrows():
        for ind in industries:
            for cycle in cycles:
                for q in quarters:
                    # Baseline log contribution influenced by ideology and party
                    latent_contrib = (
                        9.2 
                        + 0.6 * leg["dw_nominate_dim1"] * (1.0 if ind["repub_lean"] > 0 else -1.0)
                        + 0.05 * leg["tenure_years"]
                        + np.random.normal(0, 0.8)
                    )
                    amount = np.exp(latent_contrib)
                    # Occasional large PAC funding spike (treatment shock for DiD)
                    is_spike = 1 if np.random.rand() < 0.05 else 0
                    if is_spike:
                        amount *= np.random.uniform(3.0, 6.0)

                    contributions.append({
                        "contribution_id": f"CONT_{len(contributions)+1}",
                        "legislator_id": leg["legislator_id"],
                        "fec_cand_id": leg["fec_cand_id"],
                        "industry_code": ind["code"],
                        "industry_name": ind["name"],
                        "cycle": cycle,
                        "quarter": f"{cycle}_{q}",
                        "amount": round(amount, 2),
                        "is_spike": is_spike
                    })
    df_contributions = pd.DataFrame(contributions)

    # 3. Roll-Call Votes on Industry Legislation
    votes = []
    bill_id_counter = 1
    for ind in industries:
        for b_idx in range(n_bills_per_industry):
            bill_id = f"HR_{ind['code']}_{bill_id_counter}"
            bill_id_counter += 1
            bill_year = np.random.choice([2019, 2020, 2021, 2022])
            
            # Aggregate donations received by legislator from this industry prior to bill
            leg_donations = df_contributions[
                (df_contributions["industry_code"] == ind["code"]) &
                (df_contributions["cycle"] <= bill_year)
            ].groupby("legislator_id")["amount"].sum().to_dict()

            for _, leg in df_legislators.iterrows():
                don_amt = leg_donations.get(leg["legislator_id"], 1000.0)
                log_don = np.log(max(don_amt, 10.0))
                
                # Causal latent voting propensity:
                # Pro-industry vote probability increases with DW-NOMINATE alignment and donations
                latent_utility = (
                    -0.8 
                    + 2.2 * leg["dw_nominate_dim1"]
                    + 0.45 * (log_don - 9.0)
                    + 0.3 * (1 if leg["party"] == "R" else 0)
                    + np.random.logistic(0, 1.0)
                )
                vote_prob = 1.0 / (1.0 + np.exp(-latent_utility))
                voted_pro_industry = 1 if np.random.rand() < vote_prob else 0

                votes.append({
                    "vote_record_id": f"VOTE_{len(votes)+1}",
                    "bill_id": bill_id,
                    "industry_code": ind["code"],
                    "legislator_id": leg["legislator_id"],
                    "year": bill_year,
                    "quarter": f"{bill_year}_Q{np.random.choice([1, 2, 3, 4])}",
                    "voted_pro_industry": voted_pro_industry,
                    "dw_nominate_dim1": leg["dw_nominate_dim1"],
                    "prior_contributions_usd": don_amt,
                    "log_contributions": log_don
                })
    df_votes = pd.DataFrame(votes)

    return df_legislators, df_contributions, df_votes
