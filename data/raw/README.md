# Raw Data Sources & Ingestion Protocols

This directory stores raw, unmanipulated campaign finance, congressional voting, and legislative metadata files. Due to storage constraints and federal data licensing, raw bulk archives (often exceeding 60 million records and 40 GB uncompressed) are excluded from version control via `.gitignore`. Follow the instructions below to acquire the raw source files or use `scripts/data_collection.py` for automated extraction.

---

## 1. Federal Election Commission (FEC) Bulk Data

The Federal Election Commission provides biennial bulk archive bundles containing campaign committee registrations, candidate declarations, PAC contributions, and individual itemized donations.

### Required Files per Two-Year Cycle (e.g., 2016–2024):
1. **Candidate Master File (`cn.txt`)**:
   - URL: `https://www.fec.gov/files/bulk-downloads/{cycle}/cn{short_yr}.zip`
   - Description: Unique candidate identifiers (`CAND_ID`), candidate names, party affiliations, election years, and office sought (House `H`, Senate `S`, President `P`).
   - Format: Pipe-delimited (`|`), headerless. Header schema specified in `data/external/cnd_header.csv`.

2. **Committee Master File (`cm.txt`)**:
   - URL: `https://www.fec.gov/files/bulk-downloads/{cycle}/cm{short_yr}.zip`
   - Description: FEC committee identifiers (`CMTE_ID`), committee types (PAC `Q`/`N`, Super PAC `O`, Campaign Committee `H`/`S`), connected organization name, and treasurer.

3. **Candidate-to-Committee Linkage (`ccl.txt`)**:
   - URL: `https://www.fec.gov/files/bulk-downloads/{cycle}/ccl{short_yr}.zip`
   - Description: Maps principal campaign committees and authorized committees to specific Candidate IDs (`CAND_ID` <-> `CMTE_ID`).

4. **Contributions to Candidates from Committees / PACs (`pas2.txt`)**:
   - URL: `https://www.fec.gov/files/bulk-downloads/{cycle}/pas2{short_yr}.zip`
   - Description: Every direct contribution or independent expenditure made by a PAC or party committee to a candidate committee.

5. **Individual Contributions (`itcont.txt` / `indiv.zip`)**:
   - URL: `https://www.fec.gov/files/bulk-downloads/{cycle}/indiv{short_yr}.zip`
   - Description: Itemized individual donor contributions ($200+ statutory reporting threshold), including contributor names, addresses, employers, occupations, transaction dates, and transaction amounts.

---

## 2. ProPublica Congress API / Congress.gov Roll-Call Votes

Legislative roll-call voting records are retrieved via the ProPublica Congress API (or Congress.gov API v3):

- **Roll-Call Votes Feed**:
  - Endpoint: `https://api.propublica.org/congress/v1/{congress}/{chamber}/votes/recent.json`
  - Roll-Call Member Details: `https://api.propublica.org/congress/v1/{congress}/{chamber}/sessions/{session_number}/votes/{roll_call_number}.json`
- **Authentication**: Free API Key obtained at `https://www.propublica.org/datastore/api/propublica-congress-api` or `https://api.congress.gov/sign-up/`.
- **Payload Schema**:
  - `roll_call_id`, `bill_id`, `question`, `result`, `date`, `time`
  - Array of positions: `member_id` (Bioguide ID), `name`, `party`, `state`, `district`, `vote_position` (`Yes`, `No`, `Present`, `Not Voting`).

---

## 3. OpenSecrets / Center for Responsive Politics (CRP) Classifications

OpenSecrets categorizes PACs and corporate donors into standardized economic sectors and 5-character CRP industry codes:

1. **CRP Category Codes (`CRP_Categories.txt`)**:
   - Source: `https://www.opensecrets.org/downloads/crp/CRP_Categories.txt`
   - Structure: `Catcode`, `Catname`, `CatOrder`, `Industry`, `Sector`, `Sector_Long`
   - Example: `E01` (Oil & Gas) under `Energy & Natural Resources`.

2. **CRP Candidate-to-FEC Crosswalk (`cands{cycle}.txt`)**:
   - Bridges OpenSecrets candidate identifier (`CID`) with FEC Candidate ID (`CAND_ID`) and Congressional Bioguide ID (`bioguide_id`).

---

## 4. Automated Acquisition

To automatically pull sample cycles, verify checksums, and extract files into `data/raw/`:

```bash
python scripts/data_collection.py --cycles 2020 2022 --sample-size 50000 --download-votes
```
