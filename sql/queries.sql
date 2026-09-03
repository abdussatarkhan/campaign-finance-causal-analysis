-- ==============================================================================
-- Campaign Finance Causal Analysis: SQL Analytical Queries & Aggregations
-- Database Engine: PostgreSQL 14+ / DuckDB
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- 1. Top PAC Donors by Economic Sector and Election Cycle
-- ------------------------------------------------------------------------------
WITH sector_contributions AS (
    SELECT 
        c.cycle,
        t.sector_name,
        t.industry_name,
        c.cmte_id,
        c.cmte_nm,
        SUM(c.transaction_amt) AS total_donations,
        COUNT(*) AS donation_count,
        AVG(c.transaction_amt) AS avg_donation_size
    FROM fec_contributions_pas2 c
    INNER JOIN crp_category_crosswalk t 
        ON c.crp_cat_code = t.cat_code
    WHERE c.transaction_amt > 0
      AND c.memo_cd IS NULL  -- Exclude memo entries to prevent double counting
    GROUP BY 
        c.cycle,
        t.sector_name,
        t.industry_name,
        c.cmte_id,
        c.cmte_nm
)
SELECT 
    cycle,
    sector_name,
    industry_name,
    cmte_nm,
    total_donations,
    donation_count,
    ROUND(avg_donation_size, 2) AS avg_donation_size,
    DENSE_RANK() OVER (PARTITION BY cycle, sector_name ORDER BY total_donations DESC) AS sector_donor_rank
FROM sector_contributions
WHERE sector_donor_rank <= 10
ORDER BY cycle DESC, sector_name, sector_donor_rank;


-- ------------------------------------------------------------------------------
-- 2. Donor Concentration & Herfindahl-Hirschman Index (HHI) per Legislator
-- ------------------------------------------------------------------------------
WITH legislator_industry_totals AS (
    SELECT 
        l.bioguide_id,
        l.fec_cand_id,
        l.candidate_name,
        l.party,
        c.cycle,
        t.sector_name,
        SUM(c.transaction_amt) AS industry_amt
    FROM fec_contributions_pas2 c
    INNER JOIN fec_candidate_linkage ccl 
        ON c.cmte_id = ccl.cmte_id
    INNER JOIN legislator_master l 
        ON ccl.cand_id = l.fec_cand_id
    INNER JOIN crp_category_crosswalk t 
        ON c.crp_cat_code = t.cat_code
    GROUP BY 
        l.bioguide_id,
        l.fec_cand_id,
        l.candidate_name,
        l.party,
        c.cycle,
        t.sector_name
),
legislator_shares AS (
    SELECT 
        bioguide_id,
        candidate_name,
        party,
        cycle,
        sector_name,
        industry_amt,
        SUM(industry_amt) OVER (PARTITION BY bioguide_id, cycle) AS total_pac_raised,
        100.0 * industry_amt / NULLIF(SUM(industry_amt) OVER (PARTITION BY bioguide_id, cycle), 0) AS sector_share_pct
    FROM legislator_industry_totals
)
SELECT 
    bioguide_id,
    candidate_name,
    party,
    cycle,
    total_pac_raised,
    -- HHI = sum of squared market shares (ranges from near 0 to 10,000)
    ROUND(SUM(POWER(sector_share_pct, 2)), 2) AS donor_concentration_hhi,
    CASE 
        WHEN SUM(POWER(sector_share_pct, 2)) > 2500 THEN 'Highly Concentrated'
        WHEN SUM(POWER(sector_share_pct, 2)) BETWEEN 1500 AND 2500 THEN 'Moderately Concentrated'
        ELSE 'Unconcentrated / Diversified'
    END AS concentration_category
FROM legislator_shares
GROUP BY 
    bioguide_id,
    candidate_name,
    party,
    cycle,
    total_pac_raised
ORDER BY total_pac_raised DESC;


-- ------------------------------------------------------------------------------
-- 3. Close Election Discontinuity Identification (Margin < 5%)
-- ------------------------------------------------------------------------------
WITH election_margins AS (
    SELECT 
        e.election_year,
        e.state,
        e.district,
        e.winner_cand_id,
        e.winner_name,
        e.winner_party,
        e.winner_votes,
        e.runnerup_votes,
        e.total_votes,
        (CAST(e.winner_votes AS FLOAT) - e.runnerup_votes) / NULLIF(e.total_votes, 0) AS general_vote_margin
    FROM congressional_elections e
)
SELECT 
    election_year,
    state,
    district,
    winner_name,
    winner_party,
    ROUND(general_vote_margin::numeric, 4) AS vote_margin,
    CASE 
        WHEN ABS(general_vote_margin) <= 0.05 THEN 1 
        ELSE 0 
    END AS close_election_flag,
    CASE 
        WHEN general_vote_margin >= 0 THEN 1 
        ELSE 0 
    END AS rdd_treatment_assignment
FROM election_margins
WHERE ABS(general_vote_margin) <= 0.15
ORDER BY ABS(general_vote_margin) ASC;


-- ------------------------------------------------------------------------------
-- 4. Dynamic Donation Shocks (> 2 Standard Deviations) for DiD Panel
-- ------------------------------------------------------------------------------
WITH quarterly_series AS (
    SELECT 
        l.bioguide_id,
        t.sector_name,
        c.cycle,
        c.quarter,
        SUM(c.transaction_amt) AS quarterly_donations
    FROM fec_contributions_pas2 c
    INNER JOIN fec_candidate_linkage ccl ON c.cmte_id = ccl.cmte_id
    INNER JOIN legislator_master l ON ccl.cand_id = l.fec_cand_id
    INNER JOIN crp_category_crosswalk t ON c.crp_cat_code = t.cat_code
    GROUP BY l.bioguide_id, t.sector_name, c.cycle, c.quarter
),
quarterly_stats AS (
    SELECT 
        bioguide_id,
        sector_name,
        AVG(quarterly_donations) AS mean_quarterly_donations,
        STDDEV(quarterly_donations) AS std_quarterly_donations
    FROM quarterly_series
    GROUP BY bioguide_id, sector_name
)
SELECT 
    q.bioguide_id,
    q.sector_name,
    q.cycle,
    q.quarter,
    q.quarterly_donations,
    ROUND(s.mean_quarterly_donations, 2) AS baseline_mean,
    ROUND(s.std_quarterly_donations, 2) AS baseline_std,
    ROUND(((q.quarterly_donations - s.mean_quarterly_donations) / NULLIF(s.std_quarterly_donations, 0))::numeric, 2) AS z_score,
    CASE 
        WHEN (q.quarterly_donations - s.mean_quarterly_donations) / NULLIF(s.std_quarterly_donations, 0) >= 2.0 THEN 1
        ELSE 0 
    END AS is_donation_spike_treatment
FROM quarterly_series q
INNER JOIN quarterly_stats s 
    ON q.bioguide_id = s.bioguide_id 
   AND q.sector_name = s.sector_name
ORDER BY z_score DESC;


-- ------------------------------------------------------------------------------
-- 5. Pro-Industry Voting Rate by Legislator & Industry Code
-- ------------------------------------------------------------------------------
SELECT 
    v.bioguide_id,
    m.name AS legislator_name,
    m.party,
    m.dw_nominate_dim1,
    b.target_industry_code,
    COUNT(v.vote_record_id) AS total_votes_cast,
    SUM(CASE WHEN v.voted_pro_industry = 1 THEN 1 ELSE 0 END) AS pro_industry_votes,
    ROUND(AVG(v.voted_pro_industry)::numeric, 4) AS pro_industry_vote_share
FROM congressional_roll_call_votes v
INNER JOIN legislator_master m ON v.bioguide_id = m.bioguide_id
INNER JOIN bill_industry_classification b ON v.bill_id = b.bill_id
GROUP BY 
    v.bioguide_id,
    m.name,
    m.party,
    m.dw_nominate_dim1,
    b.target_industry_code
HAVING COUNT(v.vote_record_id) >= 10
ORDER BY pro_industry_vote_share DESC;
