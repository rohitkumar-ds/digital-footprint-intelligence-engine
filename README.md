# Digital Footprint Intelligence Engine

> **An analytics project that evaluates public developer activity using API-based data collection, feature engineering, statistical analysis, and interactive visualizations. It explores indicators such as GitHub contributions, repositories, and Stack Overflow reputation to analyze developer activity and growth patterns.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Statistical%20Analysis-8CAAE6?logo=scipy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Dashboard-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Developer profiles contain useful signals across platforms such as GitHub, Stack Overflow, and Kaggle. However, raw totals alone do not tell the full story.

A developer with thousands of historical contributions may currently be inactive, while another developer with smaller totals may be growing consistently.

**Developer Digital Footprint Intelligence Engine** explores this problem by combining developer activity signals into a single analytical pipeline that focuses on:

- activity growth rather than only lifetime totals,
- cross-platform developer signals,
- normalized comparisons between differently scaled metrics,
- statistical growth trajectory,
- correlation between activity velocities,
- and an interpretable **Technical Growth Score (TGS)**.

The project combines API ingestion, lightweight web scraping, feature engineering, statistical analysis, and interactive visualization in a modular Python workflow.

---

## Why I Built This

Traditional developer evaluation often relies on static snapshot metrics:

- total GitHub contributions,
- repository count,
- Stack Overflow reputation,
- profile views,
- or isolated platform statistics.

These values show **accumulation**, but not necessarily **trajectory**.

The idea behind this project was to ask a more useful question:

> **Is a developer's technical footprint actually growing, and how consistently is that growth happening across platforms?**

The engine collects current developer signals and transforms them into a 12-month analytical series. From there, it calculates velocity, normalization, correlations, a weighted growth score, and a regression-based trajectory.

---

## Key Features

- Fetches developer activity data from GitHub and Stack Overflow APIs.
- Live GitHub profile ingestion through the REST API
- GitHub contribution extraction through lightweight web scraping
- Stack Overflow reputation ingestion through the Stack Exchange API
- 12-month reproducible time-series generation
- Month-over-month activity velocity calculation
- Log transformation for skewed metrics
- Epsilon-stabilized Min-Max normalization
- Weighted **Technical Growth Score (TGS)**
- OLS-style linear growth trajectory using `scipy.stats.linregress`
- Pearson correlation analysis across platform velocities
- Interactive dark-theme Plotly dashboard
- CSV export of the processed analytical dataset

---

## System Architecture

```text
┌──────────────────────────────────────┐
│          DATA INGESTION              │
│                                      │
│  GitHub REST API                     │
│  GitHub Contribution Scraping        │
│  Stack Exchange API                  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        FEATURE ENGINEERING           │
│                                      │
│  Month-over-Month Velocity           │
│  Log Transformation                  │
│  Min-Max Normalization               │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│   STATISTICAL INTELLIGENCE           │
│                                      │
│  Technical Growth Score (TGS)        │
│  Linear Regression                   │
│  R² / p-value                        │
│  Pearson Correlations                │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│      INTERACTIVE DASHBOARD           │
│                                      │
│  TGS Trend                           │
│  Normalized Platform Growth          │
│  Velocity Correlation Heatmap        │
│  Contribution Velocity              │
└──────────────────────────────────────┘
```

---

## Modular Design

The pipeline is separated into four focused components.

| Component | Responsibility |
|---|---|
| `FootprintDataEngine` | API requests, GitHub scraping, and time-series generation |
| `FeatureProcessor` | Velocity, logarithmic transformation, and normalization |
| `IntelligenceEngine` | TGS calculation, regression analysis, and correlations |
| `DashboardBuilder` | Interactive Plotly dashboard construction |

This separation keeps data collection, transformation, analytics, and visualization independent and easier to maintain.

---

## How the Pipeline Works

### 1. Data Ingestion

The engine retrieves live baseline information from:

**GitHub REST API**
- Public repositories
- Followers

**GitHub Contributions Page**
- Contribution count extracted using regex-based HTML parsing

**Stack Exchange API**
- Stack Overflow reputation

Kaggle contest activity is represented inside the generated analytical time series.

Because the public endpoints used here do not provide a unified historical monthly dataset for every metric, the project anchors the simulation to live profile values and generates a reproducible 12-month sequence for analysis.

> **Important:** The generated history is synthetic and should be interpreted as a modeling/demo layer, not as verified historical activity.

---

### 2. Monthly Velocity

Cumulative totals can hide changes in activity.

For every metric, month-over-month velocity is calculated using:

```python
df[f"{col}_velocity"] = df[col].diff().fillna(0)
```

Conceptually:

```text
V(t) = X(t) − X(t−1)
```

where:

- $X_t$ = cumulative metric at time $t$
- $V_t$ = change from the previous observation

This makes the analysis sensitive to **momentum**, rather than only accumulated totals.

---

### 3. Log Transformation

Developer metrics can operate on very different scales and may contain long distribution tails.

The pipeline creates log-transformed features using:

```python
np.log1p(df[col])
```

Mathematically:

```text
TGS = 100 × (
    0.40 × C_norm
  + 0.20 × R_norm
  + 0.25 × S_norm
  + 0.15 × K_norm
)
```

Where:

| Symbol | Metric | Weight |
|---|---|---:|
| $C_{\text{norm}}$ | GitHub Contributions | 40% |
| $R_{\text{norm}}$ | GitHub Repositories | 20% |
| $S_{\text{norm}}$ | Stack Overflow Reputation | 25% |
| $K_{\text{norm}}$ | Kaggle Contests | 15% |

The weights are intentionally transparent rather than learned by a machine-learning model.

They represent the scoring assumptions of this prototype and can be changed for different evaluation contexts.

---

## Statistical Intelligence

### Growth Trajectory

The project applies linear regression to TGS across the generated monthly sequence:

```python
stats.linregress(x, y)
```

The analysis returns three important indicators.

### Slope

Measures the direction and magnitude of the TGS trajectory.

```text
R² = r²
```

Values closer to `1` indicate a more consistent linear relationship.

### p-value

The regression p-value is retained as a statistical significance indicator for the fitted slope.

It should be interpreted alongside the model assumptions and the fact that this project's historical sequence is synthetically generated.

---

## Cross-Platform Correlation

The engine calculates a Pearson correlation matrix across all velocity features:

```python
df[velocity_cols].corr(method="pearson")
```

This helps explore whether changes in activity across platforms move together.

For example, the analysis can reveal whether increases in GitHub activity tend to coincide with changes in Stack Overflow or Kaggle activity within the modeled series.

Correlation should not be interpreted as causation.

---

## Dashboard

The final Plotly dashboard contains four analytical panels:

1. **Technical Growth Score Trend**
2. **Normalized Platform Growth**
3. **Activity Velocity Pearson Correlation**
4. **Monthly GitHub Contribution Velocity**

The dashboard uses Plotly's dark template and interactive traces, allowing the user to inspect the generated developer trajectory visually.

---

## Engineering Challenges I Faced

These were practical issues encountered while building and debugging the project.

### 1. Artificial First-Month Velocity Spike
Using `.diff().fillna(df[col].iloc[0])` inserted the entire cumulative value into Month 1, creating an unrealistic spike that distorted the chart.

**Fix:** Changed the baseline to `.diff().fillna(0)` so velocity begins from zero.

### 2. Plotly Legend Overlap
Plotly's shared legend caused traces from unrelated subplot panels to clutter the comparison chart.

**Fix:** Disabled legends for non-comparative traces with `showlegend=False` and manually controlled legend positioning.

### 3. Zero-Activity Edge Cases
Profiles with zero activity could create invalid `np.random.randint()` bounds, while flat metrics could produce a zero Min-Max denominator.

**Fix:** Added safe sampling bounds with `max()` and stabilized normalization using `+ 1e-8`.

### 4. GitHub Contribution Extraction
Contribution data was not available through the profile API used by the project and HTML extraction can be fragile.

**Fix:** Used a browser-like User-Agent, regex extraction (`([\d,]+)\s+contributions`), and a safe fallback value when extraction fails.

---

## Insights Generated

The project is designed to explore questions such as:

- Is developer activity accelerating or plateauing?
- How quickly is the modeled technical footprint changing?
- Are activity changes consistent across multiple platforms?
- How stable is the overall growth trajectory?
- Which platform contributes most strongly to the composite score?
- Do different activity velocities move together?

The key shift is from:

> **"How much activity exists?"**

to:

> **"How is the developer's activity changing over time?"**

---

## Potential Business Value

With verified historical data and production-grade integrations, this type of pipeline could support:

### Technical Recruiting

Move beyond raw profile totals and add trajectory-oriented signals to candidate research.

### Engineering Talent Intelligence

Compare activity patterns across technical channels using a common normalized representation.

### Developer Analytics

Identify modeled periods of acceleration, stability, or plateau.

### Portfolio Intelligence

Give developers a consolidated view of technical activity spread across different platforms.

> TGS should not be used as an automated hiring decision metric. Developer quality cannot be reduced reliably to public activity alone.



## Quick Start

Update the target developer inside `main.py`:

```python
TARGET_GITHUB = "torvalds"
TARGET_STACKOVERFLOW = "22656"
```

Then run:

```bash
python main.py
```

The pipeline will:

```text
Live Profile Data
       ↓
12-Month Modeled Time Series
       ↓
Feature Engineering
       ↓
Technical Growth Score
       ↓
Regression + Correlation Analysis
       ↓
CSV Export
       ↓
Interactive Plotly Dashboard
```

The processed dataset is exported as:

```text
data.csv
```

---

## Tech Stack

| Technology | Usage |
|---|---|
| Python | Core application |
| Pandas | Data transformation |
| NumPy | Numerical operations and simulation |
| SciPy | Regression and statistical analysis |
| Plotly | Interactive dashboard |
| Requests | API and HTTP communication |
| BeautifulSoup | HTML parsing support |
| Regex | GitHub contribution extraction |

---

## Current Limitations

This project is an analytics prototype, not a production hiring system.

Current limitations include:

- historical observations are synthetically generated from live baseline values,
- GitHub contribution scraping depends on HTML structure,
- API failures currently fall back to zero rather than using retries/caching,
- TGS weights are manually defined,
- public developer activity does not represent overall engineering ability,
- correlation and regression results inherit the assumptions of the generated dataset.

Making these limitations explicit is important because statistical sophistication does not compensate for weak or synthetic source data.

---

## Future Roadmap

- [ ] GitHub GraphQL API v4 integration
- [ ] Stack Exchange OAuth authentication
- [ ] Redis caching and API rate-limit handling
- [ ] Real historical activity ingestion
- [ ] Streamlit interface for dynamic developer searches
- [ ] Automated PDF intelligence reports
- [ ] NLP-based commit message quality analysis
- [ ] Configurable TGS weighting
- [ ] Better exception handling and request retries
- [ ] Unit tests for feature engineering and scoring logic

---

## Repository Structure

```text
developer-digital-footprint-intelligence-engine/
│
├── main.py          # Complete analytics pipeline
├── data.csv         # Generated processed dataset
├── README.md        # Project documentation
├── requirements.txt # library to download        

```

---

## What This Project Demonstrates

From an engineering perspective, this project demonstrates practical experience with:

- REST API integration
- web data extraction
- defensive numerical programming
- feature engineering
- time-series transformations
- statistical regression
- correlation analysis
- composite scoring systems
- data visualization
- object-oriented Python design
- translating technical metrics into interpretable insights

---

## License

This project is available under the **MIT License**.

---

## Final Note

This project is intentionally positioned as an **analytics and statistical engineering prototype** rather than a machine-learning system.

The goal is not to claim that public activity can perfectly measure developer ability. Instead, the project demonstrates how fragmented technical signals can be collected, transformed, normalized, statistically analyzed, and presented as an interpretable analytical workflow.
