{{ config(materialized='table') }}

WITH contributions AS (
    SELECT * FROM {{ ref('stg_contributions') }}
),

votes AS (
    SELECT * FROM {{ ref('stg_votes') }}
),

legislator_industry_contributions AS (
    SELECT 
        fec_candidate_id,
        election_cycle,
        COUNT(contribution_id) AS total_donations_count,
        COUNT(DISTINCT contributor_name) AS unique_donors_count,
        SUM(transaction_amount) AS total_amount_raised,
        AVG(transaction_amount) AS avg_donation_amount,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY transaction_amount) AS median_donation_amount,
        SUM(CASE WHEN entity_type IN ('PAC', 'COM') THEN transaction_amount ELSE 0 END) AS pac_amount_raised,
        SUM(CASE WHEN entity_type = 'IND' THEN transaction_amount ELSE 0 END) AS individual_amount_raised,
        ROUND(
            (SUM(CASE WHEN entity_type IN ('PAC', 'COM') THEN transaction_amount ELSE 0 END) / 
             NULLIF(SUM(transaction_amount), 0)) * 100.0, 
            2
        ) AS pac_funding_share_pct
    FROM contributions
    GROUP BY 
        fec_candidate_id,
        election_cycle
),

legislator_voting_activity AS (
    SELECT 
        bioguide_id,
        vote_year,
        COUNT(vote_id) AS total_roll_calls_participated,
        SUM(is_affirmative_vote) AS affirmative_votes_count,
        ROUND(AVG(is_affirmative_vote)::NUMERIC, 4) AS affirmative_vote_rate,
        SUM(CASE WHEN is_abstention THEN 1 ELSE 0 END) AS abstention_count
    FROM votes
    GROUP BY 
        bioguide_id,
        vote_year
)

SELECT 
    c.fec_candidate_id,
    v.bioguide_id,
    c.election_cycle,
    v.vote_year,
    -- Financial aggregates
    c.total_amount_raised,
    c.pac_amount_raised,
    c.individual_amount_raised,
    c.pac_funding_share_pct,
    c.total_donations_count,
    c.unique_donors_count,
    ROUND(c.avg_donation_amount, 2) AS avg_donation_amount,
    ROUND(c.median_donation_amount::NUMERIC, 2) AS median_donation_amount,
    -- Legislative aggregates
    v.total_roll_calls_participated,
    v.affirmative_votes_count,
    v.affirmative_vote_rate,
    v.abstention_count,
    -- Analytical rank
    DENSE_RANK() OVER (PARTITION BY c.election_cycle ORDER BY c.total_amount_raised DESC) AS cycle_fundraising_rank
FROM legislator_industry_contributions c
FULL OUTER JOIN legislator_voting_activity v 
    ON c.election_cycle = v.vote_year
