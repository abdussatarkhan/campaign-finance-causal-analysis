# Campaign Finance Causal Policy & Legislative Voting Analysis

[![CI](https://github.com/abdussatarkhan/campaign-finance-causal-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/abdussatarkhan/campaign-finance-causal-analysis/actions)
[![Econometrics](https://img.shields.io/badge/Causal_Inference-DiD_&_RDD-0056B3?style=for-the-badge)](https://en.wikipedia.org/wiki/Difference_in_differences) [![Python](https://img.shields.io/badge/Python-Statsmodels-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.statsmodels.org/)
[![Author](https://img.shields.io/badge/Author-Abdussatar-E50914?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abdussatarkhan)

> **An econometric causal inference framework evaluating the statistical relationship between corporate PAC contributions and congressional roll-call voting alignments using Difference-in-Differences (DiD) and Regression Discontinuity Design (RDD).**

---

## 🏛️ System Architecture

```mermaid
graph TD
    FEC[FEC PAC Donations & Lobbying Records] --> Merge[Roll-Call Legislative Vote Merging]
    Merge --> DiD[Difference-in-Differences Parallel Trends Model]
    Merge --> RDD[Regression Discontinuity Close Election Thresholds]
    DiD --> CausalEffect[Estimated Policy Influence Elasticity]
    RDD --> CausalEffect
```

---

## 🌟 Key Features & Capabilities

- **Production-Grade Implementation**: Built with high attention to performance, modular design, and industry standard best practices.
- **Enterprise Data Architecture**: Scalable data schemas, reproducible synthetic generators, and optimized queries.
- **Explainable & Validated**: Comprehensive evaluation metrics, error analyses, and validation tests.
- **Comprehensive Tech Stack**: `Python` `Statsmodels` `Pandas` `Econometrics` `Causal Inference`.

---

## 📊 Visual Preview & Analysis

<div align="center">

![campaign-finance-causal-analysis preview](images/did_parallel_trends.png)

</div>

<div align="center">

[![Daily Streak](https://img.shields.io/badge/Daily%20Streak-Active%20%F0%9F%94%A5-brightgreen?style=flat-square&logo=github)](https://github.com/abdussatarkhan)
[![Master Portfolio](https://img.shields.io/badge/Portfolio-50%2B%20Enterprise%20Projects-0e75b6?style=flat-square&logo=github)](https://github.com/abdussatarkhan/abdussatarkhan)
[![Author: Abdussatar](https://img.shields.io/badge/Author-Abdussatar-24292e?style=flat-square&logo=github)](https://github.com/abdussatarkhan)

</div>


---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdussatarkhan/campaign-finance-causal-analysis.git
cd campaign-finance-causal-analysis
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

---

## 🗺️ Roadmap & Upcoming Features

- [x] Difference-in-Differences (DiD) baseline estimation
- [x] Regression Discontinuity Design (RDD) for close elections
- [ ] Synthetic Control Group expansion with state legislative donors
- [ ] Automated campaign finance PDF report generator
- [ ] Real-time FEC API streaming ingestion

---

## 👨‍💻 Author & Profile

Built and maintained by **Abdussatar** ([@abdussatarkhan](https://github.com/abdussatarkhan)).  
For technical discussions, collaboration, or queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abdus-satar-5150813b5/) or [GitHub](https://github.com/abdussatarkhan).

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.


---

<div align="center">

### 👨‍💻 Maintained by [Abdussatar (@abdussatarkhan)](https://github.com/abdussatarkhan)
Part of the **[Master Enterprise Data Analytics & AI Portfolio](https://github.com/abdussatarkhan/abdussatarkhan)**.

⭐ If you find this repository valuable, consider dropping a star! ⭐

</div>
