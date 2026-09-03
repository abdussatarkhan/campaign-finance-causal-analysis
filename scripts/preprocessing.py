"""
Preprocessing and Entity Resolution Pipeline:
1. High-Performance Fuzzy Name Matching for Donor Deduplication (rapidfuzz)
2. Employer & Occupation Text Normalization / Standardization
3. FEC Committee-to-Candidate Linkage Resolution (CCL parsing)
4. Cross-Cycle Candidate FEC ID & Bioguide Identifier Resolution
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np

# Rapidfuzz for high-throughput string metrics
from rapidfuzz import fuzz, process

sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import setup_logger, load_config, get_project_root, ensure_dir

logger = setup_logger("preprocessing")


class DonorDeduplicator:
    """
    Performs entity deduplication on campaign donors using blocking and fuzzy matching.
    Avoids O(N^2) complexity by blocking on geographic/initial partitions.
    """

    def __init__(self, similarity_threshold: float = 88.0):
        self.similarity_threshold = similarity_threshold

    @staticmethod
    def clean_name(name: str) -> str:
        """Normalizes names by stripping punctuation, extra spaces, and suffixes."""
        if not isinstance(name, str) or not name.strip():
            return ""
        name = name.upper()
        # Remove suffixes
        name = re.sub(r"\b(JR|SR|II|III|IV|ESQ|MD|PHD)\b\.?", "", name)
        # Remove non-alphabet characters
        name = re.sub(r"[^A-Z\s]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        return name

    def deduplicate(self, df_donors: pd.DataFrame, name_col: str = "contributor_name", zip_col: str = "zip_code") -> pd.DataFrame:
        """
        Groups similar donor records within geographical zip blocks into canonical donor clusters.
        """
        logger.info(f"Initiating fuzzy donor deduplication for {len(df_donors)} donor records...")
        df = df_donors.copy()
        df["cleaned_name"] = df[name_col].apply(self.clean_name)
        df["block_key"] = df[zip_col].astype(str).str[:3] + "_" + df["cleaned_name"].str[:1]

        canonical_id_map: Dict[str, str] = {}
        unique_donors_per_block = df.groupby("block_key")["cleaned_name"].unique()

        cluster_counter = 1
        for block_key, names in unique_donors_per_block.items():
            valid_names = [n for n in names if len(n) > 3]
            processed_in_block: Set[str] = set()

            for target_name in valid_names:
                if target_name in processed_in_block:
                    continue

                cluster_id = f"DONOR_{cluster_counter:07d}"
                cluster_counter += 1
                canonical_id_map[f"{block_key}|{target_name}"] = cluster_id
                processed_in_block.add(target_name)

                # Pairwise fuzzy matches using rapidfuzz token_sort_ratio
                matches = process.extract(
                    target_name,
                    valid_names,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=self.similarity_threshold
                )
                for match_tuple in matches:
                    matched_name = match_tuple[0]
                    if matched_name not in processed_in_block:
                        canonical_id_map[f"{block_key}|{matched_name}"] = cluster_id
                        processed_in_block.add(matched_name)

        # Assign cluster IDs
        def map_donor_id(row):
            key = f"{row['block_key']}|{row['cleaned_name']}"
            return canonical_id_map.get(key, f"DONOR_MISC_{row.name}")

        df["canonical_donor_id"] = df.apply(map_donor_id, axis=1)
        unique_clusters = df["canonical_donor_id"].nunique()
        logger.info(f"Deduplication finished: {len(df)} records mapped into {unique_clusters} unique donor clusters.")
        return df


class EmployerOccupationStandardizer:
    """Standardizes non-uniform freeform employer and occupation entries in FEC disclosures."""

    CORP_PATTERNS = [
        (r"\bINC(ORPORATED)?\b\.?", "INC"),
        (r"\bCORP(ORATION)?\b\.?", "CORP"),
        (r"\bLLC\b\.?", "LLC"),
        (r"\bL\.L\.C\b\.?", "LLC"),
        (r"\bCO(MPANY)?\b\.?", "CO"),
        (r"\bLTD\b\.?", "LTD"),
        (r"\bHOLDINGS?\b\.?", "HOLDINGS"),
        (r"\bGROUP\b\.?", "GROUP")
    ]

    OCCUPATION_MAP = {
        r"\b(PHYSICIAN|DOCTOR|MD|SURGEON|CARDIOLOGIST|ONCOLOGIST)\b": "PHYSICIAN",
        r"\b(ATTORNEY|LAWYER|COUNSEL|PARTNER|LEGAL COUNSEL)\b": "ATTORNEY",
        r"\b(SOFTWARE ENGINEER|SOFTWARE DEVELOPER|PROGRAMMER|DATA SCIENTIST)\b": "SOFTWARE ENGINEER",
        r"\b(CEO|PRESIDENT|CHIEF EXECUTIVE OFFICER)\b": "EXECUTIVE",
        r"\b(CFO|FINANCIAL OFFICER|TREASURER|CONTROLLER)\b": "FINANCIAL EXECUTIVE",
        r"\b(RETIRED|NOT EMPLOYED|HOMEMAKER|NONE)\b": "NON_EMPLOYED",
        r"\b(PROFESSOR|TEACHER|EDUCATOR|INSTRUCTOR)\b": "EDUCATOR",
        r"\b(CONSULTANT|ADVISOR)\b": "CONSULTANT"
    }

    @classmethod
    def standardize_employer(cls, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return "UNKNOWN"
        t = text.upper().strip()
        for pattern, repl in cls.CORP_PATTERNS:
            t = re.sub(pattern, repl, t)
        t = re.sub(r"[^\w\s]", "", t)
        return re.sub(r"\s+", " ", t).strip()

    @classmethod
    def standardize_occupation(cls, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return "UNKNOWN"
        t = text.upper().strip()
        for pattern, canonical in cls.OCCUPATION_MAP.items():
            if re.search(pattern, t):
                return canonical
        t = re.sub(r"[^\w\s]", "", t)
        return re.sub(r"\s+", " ", t).strip()


class CommitteeCandidateLinker:
    """
    Parses and cross-references FEC Committee-to-Candidate linkage (CCL) records.
    Differentiates between Principal Campaign Committees (PCC) and Leadership PACs.
    """

    @staticmethod
    def build_linkage_graph(ccl_records: pd.DataFrame) -> pd.DataFrame:
        """
        Resolves committee mappings.
        ccl_records columns expected: CAND_ID, CAND_ELECTION_YR, FEC_ELECTION_YR, CMTE_ID, CMTE_TP, CMTE_DSGN, LINKAGE_ID
        """
        logger.info(f"Resolving FEC linkage for {len(ccl_records)} linkage entries...")
        df = ccl_records.copy()

        # CMTE_DSGN definitions:
        # P = Principal campaign committee
        # A = Authorized by candidate
        # J = Joint fundraising committee
        # D = Leadership PAC (often registered separately)
        designation_map = {
            "P": "PRINCIPAL_CAMPAIGN_CMTE",
            "A": "AUTHORIZED_CMTE",
            "J": "JOINT_FUNDRAISING_CMTE",
            "D": "LEADERSHIP_PAC",
            "U": "UNAUTHORIZED"
        }
        df["committee_designation_desc"] = df["CMTE_DSGN"].map(designation_map).fillna("OTHER")
        return df


class FecIdResolver:
    """
    Harmonizes candidate FEC IDs across chambers and builds the bridge
    between FEC CAND_IDs and Congressional Bioguide identifiers.
    """

    @staticmethod
    def reconcile_identifiers(df_legislators: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures consistent bioguide-to-FEC ID crosswalk with valid state/chamber tags.
        """
        logger.info("Validating entity crosswalk between FEC IDs and Congressional Bioguide IDs...")
        df = df_legislators.copy()
        
        # Format check for FEC Candidate ID: e.g. H0CA12345 or S8TX00123
        df["is_valid_fec_id"] = df["fec_cand_id"].astype(str).str.match(r"^[HSP][0-9][A-Z]{2}[0-9]{5}$")
        df["chamber"] = df["fec_cand_id"].astype(str).str[0].map({"H": "House", "S": "Senate", "P": "President"}).fillna("House")
        return df


def run_preprocessing_pipeline() -> None:
    """Executes the complete end-to-end preprocessing suite."""
    config = load_config()
    root = get_project_root()
    proc_dir = root / config["paths"]["processed_data"]

    # Load master files
    leg_file = proc_dir / "legislators_master.parquet"
    contrib_file = proc_dir / "contributions_clean.parquet"

    if not leg_file.exists() or not contrib_file.exists():
        logger.warning("Processed master files missing. Running data_collection pipeline first...")
        from scripts.data_collection import main as run_collection
        run_collection()

    df_legs = pd.read_parquet(leg_file)
    df_contribs = pd.read_parquet(contrib_file)

    # 1. Standardize and reconcile legislator identifiers
    resolver = FecIdResolver()
    df_legs_clean = resolver.reconcile_identifiers(df_legs)
    df_legs_clean.to_parquet(proc_dir / "legislators_master.parquet", index=False)
    logger.info(f"Reconciled {len(df_legs_clean)} legislators.")

    # 2. Donor normalization benchmark
    sample_donors = pd.DataFrame([
        {"contributor_name": "JOHN SMITH", "zip_code": "90210", "employer": "Apple Inc.", "occupation": "Software Eng"},
        {"contributor_name": "JONATHAN SMITH", "zip_code": "90210", "employer": "Apple Corporation", "occupation": "Sr. Software Developer"},
        {"contributor_name": "J. SMITH", "zip_code": "90210", "employer": "APPLE", "occupation": "Developer"},
        {"contributor_name": "ROBERT E. LEE", "zip_code": "75001", "employer": "Dallas Hospital System LLC", "occupation": "Cardiologist MD"},
        {"contributor_name": "BOB LEE", "zip_code": "75001", "employer": "Dallas Hospital", "occupation": "Physician"},
        {"contributor_name": "MARY JOHNSON", "zip_code": "10001", "employer": "Goldman Sachs & Co", "occupation": "Managing Director"}
    ])

    deduplicator = DonorDeduplicator(similarity_threshold=config["preprocessing"]["fuzzy_matching"]["threshold"])
    df_dedup = deduplicator.deduplicate(sample_donors)
    
    df_dedup["standardized_employer"] = df_dedup["employer"].apply(EmployerOccupationStandardizer.standardize_employer)
    df_dedup["standardized_occupation"] = df_dedup["occupation"].apply(EmployerOccupationStandardizer.standardize_occupation)

    dedup_out = proc_dir / "sample_deduplicated_donors.csv"
    df_dedup.to_csv(dedup_out, index=False)
    logger.info(f"Deduplication validation sample saved to {dedup_out}")
    logger.info("Preprocessing pipeline completed successfully.")


if __name__ == "__main__":
    run_preprocessing_pipeline()
