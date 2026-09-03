# Interactive Tableau Public Dashboard Specification
**Project: Campaign Donation Influence & Legislative Voting Analyzer**

This directory details the visual architecture, data dictionary, calculated fields, and interactive layout for the Tableau Public analytical suite tracking federal campaign finance flows and congressional voting behavior.

---

## Dashboard Architecture

The Tableau workbook is organized into four interactive views designed for public policy researchers, journalists, and legislative analysts:

```
+---------------------------------------------------------------------------------------+
|                 CAMPAIGN FINANCE & LEGISLATIVE INFLUENCE ANALYZER                    |
+---------------------------------------------------------------------------------------+
| [ View 1: Geographic Flows ]  [ View 2: RDD Discontinuity ]  [ View 3: DiD Shocks ]  |
|                                                                                       |
| +-----------------------------------------------+ +---------------------------------+ |
| | Map: PAC Receipts by Congressional District   | | Top 10 PAC Donors by Industry   | |
| | (Choropleth color-coded by PAC Intensity)     | | (Bar chart with cycle filter)   | |
| +-----------------------------------------------+ +---------------------------------+ |
|                                                                                       |
| +-----------------------------------------------+ +---------------------------------+ |
| | Quadrant: DW-NOMINATE Ideology vs. Donations  | | Pro-Industry Vote Rate by Party | |
| | (Scatter plot with partisan color coding)     | | (Comparison grouped bar chart)  | |
| +-----------------------------------------------+ +---------------------------------+ |
+---------------------------------------------------------------------------------------+
```

---

## 1. View Specifications

### View 1: Geographic & Industrial Concentration Map
- **Visual Type**: Geospatial Choropleth (US Congressional Districts shapefile).
- **Measures**:
  - Color: `SUM([Total_Industry_Donations])` using diverging Red-Blue (or Gold-Purple) color palette.
  - Size/Label: Percentile rank within congress.
- **Filters**:
  - `Industry Sector` (Oil & Gas, Commercial Banks, Pharma, Defense).
  - `Election Cycle` (2016, 2018, 2020, 2022, 2024).
  - `Chamber` (House of Representatives, Senate).
- **Tooltip**: Legislator name, party, district, total industry PAC funds raised, top 3 corporate donors, pro-industry voting score.

### View 2: Causal Regression Discontinuity Explorer
- **Visual Type**: Dual-Axis Local Regression Scatter.
- **X-Axis**: `General Election Vote Margin` (Centered at 0.0).
- **Y-Axis**: `ln(Campaign Receipts)` or `Pro-Industry Roll-Call Vote Probability`.
- **Interactive Controls**:
  - Bandwidth Slider parameter `[Bandwidth_h]` (Range: 0.02 to 0.15, step: 0.01).
  - Kernel Selection dropdown (Triangular vs Epanechnikov).
- **Annotations**: Visual discontinuity gap at $x = 0$ displaying estimated causal jump $\hat{\tau}$, standard error, and $p$-value.

### View 3: Dynamic Event Study & Donation Shock Tracker
- **Visual Type**: Longitudinal Line Plot with Shaded Error Bands (95% CI).
- **X-Axis**: Relative Event Time $t \in [-3, +3]$ years around $>2\sigma$ PAC contribution shock.
- **Y-Axis**: Cumulative differential voting alignment ($\Delta \text{Pr}(\text{Pro-Industry Vote})$).
- **Reference Lines**: Event shock vertical marker at $t = 0$; pre-treatment zero baseline.

### View 4: Ideological Quadrant Explorer
- **Visual Type**: 2x2 Quadrant Scatter Plot.
- **X-Axis**: `DW-NOMINATE Dimension 1` (-1.0 Liberal to +1.0 Conservative).
- **Y-Axis**: `ln(Industry PAC Contributions)`.
- **Quadrants**:
  - Top-Left: Liberal High-PAC Recipients
  - Top-Right: Conservative High-PAC Recipients
  - Bottom-Left: Liberal Low-PAC Recipients
  - Bottom-Right: Conservative Low-PAC Recipients

---

## 2. Tableau Calculated Fields

Below are the core Tableau formulas implemented in the workbook:

1. **Vote Margin Centered**:
   ```tableau
   [General_Vote_Margin] - 0.0
   ```

2. **In Bandwidth Window**:
   ```tableau
   ABS([Vote_Margin_Centered]) <= [Parameters].[Bandwidth_h]
   ```

3. **Log Contributions**:
   ```tableau
   LN([Total_Industry_Donations] + 1.0)
   ```

4. **Party Color Encoding**:
   ```tableau
   IF [Party] = "D" THEN "#2b5c8f"
   ELSEIF [Party] = "R" THEN "#d95f02"
   ELSE "#7570b3"
   END
   ```

5. **Donation Shock Indicator**:
   ```tableau
   IF [Donation_Z_Score] >= 2.0 THEN "Shock (>2σ)"
   ELSE "Normal Flow"
   END
   ```

---

## 3. Data Source Connection

- **Primary Extract**: Exported from dbt mart `mart_legislator_contributions` or `data/processed/panel_analysis_master.parquet` converted to `.hyper` format.
- **Refresh Frequency**: Biennial following FEC general election certification cycles.
