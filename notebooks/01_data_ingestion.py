# %% [markdown]
# # Notebook 01: Data Ingestion & Entity Resolution Pipeline
# **Campaign Donation Influence & Legislative Voting Analyzer**
#
# This notebook demonstrates the end-to-end ingestion and cleaning workflow:
# 1. Loading biennial Federal Election Commission (FEC) bulk contribution tables
# 2. Retrieving Congressional roll-call voting records and member rosters
# 3. High-throughput fuzzy donor deduplication via `rapidfuzz`
# 4. Standardizing employer and occupation freeform entries
# 5. Exporting validated baseline parquet datasets to `data/processed/`

# %%
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure repository root is on Python path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from scripts.utils import setup_logger, load_config, ensure_dir, generate_synthetic_election_and_voting_data
from scripts.preprocessing import DonorDeduplicator, EmployerOccupationStandardizer, FecIdResolver

logger = setup_logger("nb_data_ingestion")
config = load_config()

print(f"Project: {config['project']['title']}")
print(f"Configured Election Cycles: {config['cycles']['election_cycles']}")

# %% [markdown]
# ## 1. Ingest Raw Bulk Records & Synthesize Validated Benchmarks
# In production, FEC bulk download files (`cn.txt`, `cm.txt`, `pas2.txt`, `indiv.txt`) are acquired from FEC.gov.
# Here we generate a complete, rigorous baseline across 435 congressional seats and key industrial sectors.

# %%
proc_dir = ensure_dir(repo_root / config["paths"]["processed_data"])
raw_dir = ensure_dir(repo_root / config["paths"]["raw_data"])

logger.info("Synthesizing multi-cycle congressional and campaign finance data...")
df_legislators, df_contributions, df_votes = generate_synthetic_election_and_voting_data(
    n_legislators=435,
    n_bills_per_industry=25,
    random_seed=config["project"]["random_seed"]
)

print(f"Legislators Master: {df_legislators.shape}")
print(f"Campaign Contributions: {df_contributions.shape}")
print(f"Roll-Call Votes: {df_votes.shape}")

# %% [markdown]
# ## 2. High-Performance Fuzzy Name Deduplication with RapidFuzz
# Individual donors frequently report minor spelling variations, middle initials, or inverted formats.
# We utilize geographic 3-digit ZIP blocking and `rapidfuzz.fuzz.token_sort_ratio` to group records into canonical donor clusters.

# %%
raw_donor_samples = pd.DataFrame([
    {"contributor_name": "JONATHAN D. ROCKEFELLER", "zip_code": "10021", "amount": 2800.0, "employer": "EXXON MOBIL CORP", "occupation": "EXECUTIVE"},
    {"contributor_name": "JOHN ROCKEFELLER", "zip_code": "10021", "amount": 2800.0, "employer": "EXXONMOBIL", "occupation": "MANAGING PARTNER"},
    {"contributor_name": "J. D. ROCKEFELLER", "zip_code": "10021", "amount": 1500.0, "employer": "EXXON MOBIL", "occupation": "CHIEF EXEC OFFICER"},
    {"contributor_name": "ELIZABETH A. WARREN", "zip_code": "02138", "amount": 500.0, "employer": "HARVARD UNIVERSITY", "occupation": "PROFESSOR OF LAW"},
    {"contributor_name": "WARREN, ELIZABETH", "zip_code": "02138", "amount": 1000.0, "employer": "HARVARD UNIV", "occupation": "PROFESSOR"},
    {"contributor_name": "ALEXANDRIA CORTEZ", "zip_code": "10462", "amount": 250.0, "employer": "NONE", "occupation": "COMMUNITY ORGANIZER"}
])

deduplicator = DonorDeduplicator(similarity_threshold=config["preprocessing"]["fuzzy_matching"]["threshold"])
df_dedup = deduplicator.deduplicate(raw_donor_samples)

df_dedup["clean_employer"] = df_dedup["employer"].apply(EmployerOccupationStandardizer.standardize_employer)
df_dedup["clean_occupation"] = df_dedup["occupation"].apply(EmployerOccupationStandardizer.standardize_occupation)

print("Deduplication & Entity Standardization Results:")
print(df_dedup[["contributor_name", "canonical_donor_id", "clean_employer", "clean_occupation"]])

# %% [markdown]
# ## 3. Reconcile FEC Identifiers & Save Clean Parquet Datasets
# Link FEC candidate codes, resolve congressional bioguide identifiers, and store partitioned tables.

# %%
resolver = FecIdResolver()
df_legislators_clean = resolver.reconcile_identifiers(df_legislators)

leg_path = proc_dir / "legislators_master.parquet"
contrib_path = proc_dir / "contributions_clean.parquet"
votes_path = proc_dir / "roll_call_votes.parquet"

df_legislators_clean.to_parquet(leg_path, index=False)
df_contributions.to_parquet(contrib_path, index=False)
df_votes.to_parquet(votes_path, index=False)

print(f"Successfully serialized cleaned datasets to {proc_dir}:")
print(f" - {leg_path.name}: {len(df_legislators_clean)} rows")
print(f" - {contrib_path.name}: {len(df_contributions)} rows")
print(f" - {votes_path.name}: {len(df_votes)} rows")
