{{ config(materialized='view') }}

WITH raw_contributions AS (
    SELECT 
        sub_id AS contribution_id,
        cmte_id AS committee_id,
        cand_id AS fec_candidate_id,
        entity_tp AS entity_type,
        name AS contributor_name,
        city AS contributor_city,
        state AS contributor_state,
        zip_code AS contributor_zip,
        employer AS contributor_employer,
        occupation AS contributor_occupation,
        transaction_dt AS raw_transaction_date,
        transaction_amt AS raw_amount,
        memo_cd AS memo_code,
        memo_text,
        file_num,
        tran_id AS transaction_id
    FROM {{ source('fec_raw', 'raw_contributions') }}
),

cleaned_contributions AS (
    SELECT 
        contribution_id,
        committee_id,
        fec_candidate_id,
        COALESCE(entity_type, 'IND') AS entity_type,
        UPPER(TRIM(contributor_name)) AS contributor_name,
        UPPER(TRIM(contributor_city)) AS contributor_city,
        UPPER(TRIM(contributor_state)) AS contributor_state,
        SUBSTRING(REGEXP_REPLACE(contributor_zip, '[^0-9]', '', 'g'), 1, 5) AS contributor_zip5,
        UPPER(TRIM(contributor_employer)) AS contributor_employer,
        UPPER(TRIM(contributor_occupation)) AS contributor_occupation,
        TO_DATE(raw_transaction_date, 'MMDDYYYY') AS transaction_date,
        CAST(raw_amount AS NUMERIC(12, 2)) AS transaction_amount,
        CASE 
            WHEN raw_amount < 0 THEN TRUE 
            ELSE FALSE 
        END AS is_refund,
        memo_code
    FROM raw_contributions
    WHERE raw_amount IS NOT NULL
      AND (memo_code IS NULL OR memo_code != 'X') -- Filter out duplicate memo records
)

SELECT 
    contribution_id,
    committee_id,
    fec_candidate_id,
    entity_type,
    contributor_name,
    contributor_city,
    contributor_state,
    contributor_zip5,
    contributor_employer,
    contributor_occupation,
    transaction_date,
    transaction_amount,
    EXTRACT(YEAR FROM transaction_date) AS transaction_year,
    -- Map year to 2-year federal election cycle
    CASE 
        WHEN EXTRACT(YEAR FROM transaction_date)::INTEGER % 2 = 1 
        THEN EXTRACT(YEAR FROM transaction_date)::INTEGER + 1
        ELSE EXTRACT(YEAR FROM transaction_date)::INTEGER
    END AS election_cycle,
    is_refund
FROM cleaned_contributions
WHERE NOT is_refund
  AND transaction_amount >= 50.00
