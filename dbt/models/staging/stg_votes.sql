{{ config(materialized='view') }}

WITH raw_votes AS (
    SELECT 
        vote_id,
        roll_call_id,
        bill_id,
        congress,
        session,
        chamber,
        member_id AS bioguide_id,
        vote_position,
        vote_date,
        bill_title,
        question,
        result
    FROM {{ source('congress_raw', 'raw_roll_call_votes') }}
),

standardized_votes AS (
    SELECT 
        vote_id,
        roll_call_id,
        bill_id,
        CAST(congress AS INTEGER) AS congress,
        CAST(session AS INTEGER) AS session_number,
        LOWER(TRIM(chamber)) AS chamber,
        UPPER(TRIM(bioguide_id)) AS bioguide_id,
        UPPER(TRIM(vote_position)) AS vote_position_raw,
        CAST(vote_date AS TIMESTAMP) AS vote_timestamp,
        DATE(vote_date) AS vote_date,
        EXTRACT(YEAR FROM CAST(vote_date AS TIMESTAMP)) AS vote_year,
        bill_title,
        question,
        result
    FROM raw_votes
)

SELECT 
    vote_id,
    roll_call_id,
    bill_id,
    congress,
    session_number,
    chamber,
    bioguide_id,
    vote_date,
    vote_year,
    vote_position_raw,
    CASE 
        WHEN vote_position_raw IN ('YES', 'AYE', 'YEA') THEN 1
        WHEN vote_position_raw IN ('NO', 'NAY') THEN 0
        ELSE NULL 
    END AS is_affirmative_vote,
    CASE 
        WHEN vote_position_raw IN ('NOT VOTING', 'PRESENT', 'SPEAKER') THEN TRUE 
        ELSE FALSE 
    END AS is_abstention,
    bill_title,
    question,
    result
FROM standardized_votes
