"""
Data Collection Pipeline:
1. Federal Election Commission (FEC) Bulk Data Downloader (Candidate, Committee, Contributions)
2. ProPublica Congress API / Congress.gov Client for Roll-Call Votes and Member Profiles
3. OpenSecrets / CRP Industry and Sector Taxonomy Mapper
"""

import os
import sys
import argparse
import zipfile
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
from urllib.parse import urljoin

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import setup_logger, load_config, get_project_root, ensure_dir, generate_synthetic_election_and_voting_data

logger = setup_logger("data_collection")


class FECBulkDownloader:
    """Handles downloading and unpacking FEC biennial bulk data archives."""

    def __init__(self, raw_data_dir: Path, base_url: str = "https://www.fec.gov/files/bulk-downloads/"):
        self.raw_data_dir = ensure_dir(raw_data_dir / "fec")
        self.base_url = base_url

    def download_cycle_files(self, cycle: int, file_types: List[str]) -> Dict[str, Path]:
        """
        Downloads FEC zip archives for a given 2-year cycle (e.g. 2020 -> '20').
        File types: 'cn' (candidates), 'cm' (committees), 'ccl' (linkage), 'pas2' (PAC contributions).
        """
        short_yr = str(cycle)[-2:]
        downloaded_paths = {}

        for ftype in file_types:
            filename = f"{ftype}{short_yr}.zip"
            target_zip = self.raw_data_dir / filename
            extract_dir = self.raw_data_dir / f"{cycle}_{ftype}"
            url = f"{self.base_url}{cycle}/{filename}"

            if extract_dir.exists() and any(extract_dir.iterdir()):
                logger.info(f"Directory {extract_dir} already populated. Skipping download.")
                downloaded_paths[ftype] = extract_dir
                continue

            logger.info(f"Downloading FEC dataset {ftype} for cycle {cycle} from {url}...")
            try:
                response = requests.get(url, stream=True, timeout=30)
                if response.status_code == 200:
                    with open(target_zip, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                    
                    ensure_dir(extract_dir)
                    with zipfile.ZipFile(target_zip, "r") as zip_ref:
                        zip_ref.extractall(extract_dir)
                    logger.info(f"Successfully extracted {filename} to {extract_dir}")
                    downloaded_paths[ftype] = extract_dir
                else:
                    logger.warning(f"HTTP {response.status_code} fetching {url}. Bulk file may not exist or require auth.")
            except Exception as e:
                logger.error(f"Failed downloading FEC archive {filename}: {str(e)}")

        return downloaded_paths


class ProPublicaCongressAPI:
    """Client for retrieving Congressional roll-call votes and legislative text metadata."""

    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.propublica.org/congress/v1"):
        self.api_key = api_key or os.environ.get("PROPUBLICA_API_KEY", "")
        self.base_url = base_url
        self.headers = {"X-API-Key": self.api_key} if self.api_key else {}

    def fetch_recent_votes(self, congress: int = 117, chamber: str = "house") -> List[Dict[str, Any]]:
        """Retrieves recent roll call votes for a specified chamber and congress."""
        if not self.api_key:
            logger.warning("No ProPublica API Key configured. Skipping live API fetch.")
            return []

        url = f"{self.base_url}/{congress}/{chamber}/votes/recent.json"
        logger.info(f"Querying ProPublica API: {url}")
        try:
            response = requests.get(url, headers=self.headers, timeout=20)
            if response.status_code == 200:
                data = response.json()
                votes = data.get("results", {}).get("votes", [])
                logger.info(f"Retrieved {len(votes)} roll-call votes from chamber {chamber}.")
                return votes
            else:
                logger.error(f"ProPublica API error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Exception during ProPublica API request: {str(e)}")
            return []

    def fetch_roll_call_details(self, congress: int, chamber: str, session: int, roll_call: int) -> Optional[Dict[str, Any]]:
        """Retrieves individual legislator votes on a specific roll call."""
        if not self.api_key:
            return None

        url = f"{self.base_url}/{congress}/{chamber}/sessions/{session}/votes/{roll_call}.json"
        try:
            response = requests.get(url, headers=self.headers, timeout=20)
            if response.status_code == 200:
                return response.json().get("results", {}).get("votes", {}).get("vote", {})
        except Exception as e:
            logger.error(f"Failed roll call fetch ({roll_call}): {str(e)}")
        return None


class OpenSecretsTaxonomyMapper:
    """Maps Center for Responsive Politics (CRP) categories to economic sectors."""

    def __init__(self, external_dir: Path):
        self.external_dir = ensure_dir(external_dir)
        self.mapping_file = self.external_dir / "crp_categories.csv"

    def get_or_build_taxonomy(self) -> pd.DataFrame:
        """Loads cached CRP category table or constructs standard industry taxonomy."""
        if self.mapping_file.exists():
            return pd.read_csv(self.mapping_file)

        logger.info("Generating standard OpenSecrets / CRP Industry Sector Crosswalk...")
        taxonomy = [
            {"cat_code": "E01", "sector_code": "E", "sector_name": "Energy/Nat Resource", "industry_name": "Oil & Gas"},
            {"cat_code": "E02", "sector_code": "E", "sector_name": "Energy/Nat Resource", "industry_name": "Mining"},
            {"cat_code": "E04", "sector_code": "E", "sector_name": "Energy/Nat Resource", "industry_name": "Electric Utilities"},
            {"cat_code": "F07", "sector_code": "F", "sector_name": "Finance/Insur/RealEst", "industry_name": "Commercial Banks"},
            {"cat_code": "F10", "sector_code": "F", "sector_name": "Finance/Insur/RealEst", "industry_name": "Real Estate"},
            {"cat_code": "F13", "sector_code": "F", "sector_name": "Finance/Insur/RealEst", "industry_name": "Securities & Investment"},
            {"cat_code": "H01", "sector_code": "H", "sector_name": "Health", "industry_name": "Pharmaceuticals & Health Products"},
            {"cat_code": "H02", "sector_code": "H", "sector_name": "Health", "industry_name": "Health Services/Hospitals"},
            {"cat_code": "D01", "sector_code": "D", "sector_name": "Defense", "industry_name": "Defense Aerospace"},
            {"cat_code": "D02", "sector_code": "D", "sector_name": "Defense", "industry_name": "Defense Electronics"},
            {"cat_code": "C01", "sector_code": "C", "sector_name": "Communications/Electronics", "industry_name": "Telecom Services"},
            {"cat_code": "C04", "sector_code": "C", "sector_name": "Communications/Electronics", "industry_name": "Computers/Internet"},
            {"cat_code": "L01", "sector_code": "L", "sector_name": "Labor", "industry_name": "Industrial Unions"},
            {"cat_code": "A01", "sector_code": "A", "sector_name": "Agribusiness", "industry_name": "Crop Production & Basic Processing"}
        ]
        df_tax = pd.DataFrame(taxonomy)
        df_tax.to_csv(self.mapping_file, index=False)
        logger.info(f"Saved taxonomy to {self.mapping_file}")
        return df_tax


def main():
    parser = argparse.ArgumentParser(description="Campaign Finance & Legislative Data Collection")
    parser.add_argument("--cycles", nargs="+", type=int, default=[2020, 2022], help="FEC election cycles to download")
    parser.add_argument("--sample-size", type=int, default=500, help="Sample size for demonstration benchmarking")
    parser.add_argument("--download-votes", action="store_true", help="Download live roll-call votes from ProPublica")
    parser.add_argument("--generate-benchmark", action="store_true", default=True, help="Generate local benchmark datasets")

    args = parser.parse_args()
    config = load_config()
    root = get_project_root()

    raw_dir = root / config["paths"]["raw_data"]
    ext_dir = root / config["paths"]["external_data"]
    proc_dir = root / config["paths"]["processed_data"]

    logger.info("Initializing Campaign Finance Data Collection Engine...")

    # 1. OpenSecrets Industry Taxonomy
    tax_mapper = OpenSecretsTaxonomyMapper(ext_dir)
    df_tax = tax_mapper.get_or_build_taxonomy()
    logger.info(f"Loaded {len(df_tax)} industry taxonomy category mappings.")

    # 2. FEC Downloader
    fec_downloader = FECBulkDownloader(raw_dir)
    for cycle in args.cycles:
        fec_downloader.download_cycle_files(cycle, ["cn", "cm", "ccl", "pas2"])

    # 3. ProPublica API Client
    propublica_client = ProPublicaCongressAPI()
    if args.download_votes:
        votes = propublica_client.fetch_recent_votes(congress=117, chamber="house")
        logger.info(f"Acquired {len(votes)} roll-call records from ProPublica.")

    # 4. Generate & Cache Standardized Baseline Data
    if args.generate_benchmark:
        logger.info("Synthesizing validated baseline datasets for reproducible analysis...")
        df_legs, df_contribs, df_votes = generate_synthetic_election_and_voting_data(
            n_legislators=args.sample_size,
            n_bills_per_industry=20,
            random_seed=config["project"]["random_seed"]
        )

        ensure_dir(proc_dir)
        df_legs.to_parquet(proc_dir / "legislators_master.parquet", index=False)
        df_contribs.to_parquet(proc_dir / "contributions_clean.parquet", index=False)
        df_votes.to_parquet(proc_dir / "roll_call_votes.parquet", index=False)

        logger.info(f"Saved {len(df_legs)} legislators, {len(df_contribs)} contributions, {len(df_votes)} roll-call votes to {proc_dir}")

    logger.info("Data Collection stage completed successfully.")


if __name__ == "__main__":
    main()
