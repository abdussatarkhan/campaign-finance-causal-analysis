"""
Data Integration and Econometric Panel Builder:
1. Merges Campaign Contributions to Legislators via FEC Candidate IDs
2. Joins Legislator Profiles & Ideological Scores (DW-NOMINATE) to Roll-Call Votes
3. Classifies Congressional Bills by Target Industry using Committee Jurisdiction & NLP Keywords
4. Constructs the Unified Panel Dataset for Causal Inference Estimators
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import setup_logger, load_config, get_project_root, ensure_dir

logger = setup_logger("data_integration")


class BillIndustryClassifier:
    """Classifies legislative bills into target industries using committee assignment and keyword matching."""

    def __init__(self, industry_configs: List[Dict[str, Any]]):
        self.industry_configs = industry_configs

    def classify_bill(self, bill_title: str, committee_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Evaluates bill text and jurisdiction committee against industry taxonomies.
        Returns matched industry codes and confidence score.
        """
        matches = []
        clean_text = str(bill_title).lower()

        for ind in self.industry_configs:
            keyword_hits = sum(1 for kw in ind.get("keywords", []) if kw.lower() in clean_text)
            committee_match = 1 if committee_code and committee_code in ind.get("key_committees", []) else 0

            if keyword_hits > 0 or committee_match:
                score = (keyword_hits * 0.4) + (committee_match * 0.6)
                matches.append({
                    "industry_code": ind["code"],
                    "industry_name": ind["name"],
                    "confidence_score": min(score, 1.0)
                })
        return matches


class PanelDataIntegrator:
    """
    Constructs the longitudinal panel bridging legislator characteristics,
    industry PAC cash flows, and roll-call legislative votes.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.root = get_project_root()
        self.proc_dir = ensure_dir(self.root / config["paths"]["processed_data"])

    def build_integrated_panel(self) -> pd.DataFrame:
        """
        Executes relational joins, computes time-lagged contribution totals,
        and constructs econometric treatment indicators.
        """
        logger.info("Loading cleaned component datasets...")
        leg_file = self.proc_dir / "legislators_master.parquet"
        contrib_file = self.proc_dir / "contributions_clean.parquet"
        votes_file = self.proc_dir / "roll_call_votes.parquet"

        if not all(p.exists() for p in [leg_file, contrib_file, votes_file]):
            raise FileNotFoundError("Prerequisite parquet files missing in data/processed/. Run preprocessing first.")

        df_legs = pd.read_parquet(leg_file)
        df_contribs = pd.read_parquet(contrib_file)
        df_votes = pd.read_parquet(votes_file)

        logger.info(f"Loaded: {len(df_legs)} legislators, {len(df_contribs)} contributions, {len(df_votes)} roll-call votes.")

        # 1. Aggregate Contributions by Legislator, Industry, and Cycle
        logger.info("Aggregating campaign contributions across industry cycles...")
        df_contrib_agg = df_contribs.groupby(
            ["legislator_id", "industry_code"]
        ).agg(
            total_industry_donations=("amount", "sum"),
            avg_donation_size=("amount", "mean"),
            spike_count=("is_spike", "sum"),
            num_contributions=("amount", "count")
        ).reset_index()

        # Compute industry-level donor percentiles and treatment thresholds
        treatment_thresholds = df_contrib_agg.groupby("industry_code")["total_industry_donations"].quantile(
            self.config["causal_inference"]["propensity_score_matching"]["treatment_percentile_cutoff"]
        ).to_dict()

        df_contrib_agg["industry_donation_75th_pct"] = df_contrib_agg["industry_code"].map(treatment_thresholds)
        df_contrib_agg["high_donation_treatment"] = (
            df_contrib_agg["total_industry_donations"] >= df_contrib_agg["industry_donation_75th_pct"]
        ).astype(int)

        # 2. Join Roll-Call Votes with Legislators
        logger.info("Merging roll-call records with legislator ideological covariates...")
        df_panel = df_votes.merge(
            df_legs,
            on="legislator_id",
            how="inner",
            suffixes=("", "_leg")
        )

        # 3. Join with Aggregated Industry Contributions
        logger.info("Joining cumulative industry cash flows to voting records...")
        df_panel = df_panel.merge(
            df_contrib_agg,
            on=["legislator_id", "industry_code"],
            how="left"
        )

        # Fill missing donations with 0
        df_panel["total_industry_donations"] = df_panel["total_industry_donations"].fillna(0.0)
        df_panel["high_donation_treatment"] = df_panel["high_donation_treatment"].fillna(0).astype(int)
        df_panel["spike_count"] = df_panel["spike_count"].fillna(0).astype(int)
        df_panel["log_donations"] = np.log1p(df_panel["total_industry_donations"])

        # 4. Construct Causal Indicators
        # RDD Treatment: Did the candidate win general election in a razor-thin margin?
        rdd_margin = self.config["causal_inference"]["regression_discontinuity"]["close_election_margin"]
        df_panel["close_election_flag"] = (df_panel["vote_margin"].abs() <= rdd_margin).astype(int)
        df_panel["rdd_treatment_winner"] = (df_panel["vote_margin"] >= 0.0).astype(int)

        # DiD Indicator: Presence of donation spike (>2 sigma) in prior period
        df_panel["did_treated_unit"] = (df_panel["spike_count"] > 0).astype(int)
        # Mock post-treatment indicator for demonstration event study
        df_panel["did_post_period"] = (df_panel["year"] >= 2021).astype(int)
        df_panel["did_interaction"] = df_panel["did_treated_unit"] * df_panel["did_post_period"]

        # Ensure correct datatypes
        df_panel["voted_pro_industry"] = df_panel["voted_pro_industry"].astype(int)
        df_panel["party_republican"] = df_panel["party_republican"].astype(int)

        output_path = self.proc_dir / "panel_analysis_master.parquet"
        df_panel.to_parquet(output_path, index=False)
        logger.info(f"Integrated econometric panel created successfully with {len(df_panel)} records and {df_panel.shape[1]} features.")
        logger.info(f"Master panel saved to: {output_path}")

        # Summary of balance
        summary_stats = df_panel.groupby(["industry_code", "high_donation_treatment"]).agg(
            n_votes=("vote_record_id", "count"),
            pro_industry_rate=("voted_pro_industry", "mean"),
            avg_dw_nominate=("dw_nominate_dim1", "mean"),
            avg_donations=("total_industry_donations", "mean")
        )
        logger.info(f"Panel Summary by Treatment Group:\n{summary_stats}")
        return df_panel


def main():
    config = load_config()
    integrator = PanelDataIntegrator(config)
    integrator.build_integrated_panel()


if __name__ == "__main__":
    main()
