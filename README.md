# Forecasting University Enrollment Demand: IPEDS Data Analysis (2010-2021)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LaTeX](https://img.shields.io/badge/LaTeX-PDF-green.svg)](https://www.latex-project.org/)

**Bachelor Thesis**  
**Author:** Ashithosh Nithin  
**Institution:** Riga Nordic University  
**Supervisor:** Andrejs Bondarenko  
**Submission Date:** January 2026

---

## 📋 Table of Contents

- [Abstract](#abstract)
- [Research Questions](#research-questions)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## 📖 Abstract

This thesis examines enrollment forecasting methods for U.S. higher education institutions using comprehensive IPEDS (Integrated Postsecondary Education Data System) data spanning 2010-2021. The study addresses two primary research questions: (1) Which forecasting methods achieve the highest accuracy using walk-forward validation? and (2) Which institutional and affordability factors are statistically associated with enrollment demand?

**Main Findings:**
- Enrollment exhibits extremely high persistence (98% of variance explained by lagged enrollment)
- Simple naive persistence models outperform complex machine learning approaches
- Admissions funnel metrics and affordability indicators show statistical significance but limited practical impact
- COVID-19 pandemic serves as a stress test, with simple models proving most resilient

**Implications:** The findings challenge the prevailing enthusiasm for machine learning in enrollment forecasting and demonstrate that simplicity often beats complexity when processes are highly persistent.

---

## 🎯 Research Questions

### RQ1: Forecasting Model Performance
**Which forecasting methods give the most precise enrollment forecasts in a walk-forward validation setting?**

Hypothesis 1: Complex forecasting models (ARIMA, Ridge regression, Random Forest) will outperform the naive persistence baseline by at least 10% in mean absolute error.

**Result:** REJECTED  
Naive persistence achieved average MAE of 39.43 students vs. 40.33 (Ridge), 41.21 (Random Forest), and 45.87 (Moving Average).

### RQ2: Enrollment Drivers
**What institutional and affordability factors are statistically associated with enrollment demand that is stable over time even after accounting for persistence?**

Hypothesis 2: Admissions funnel metrics and affordability indicators will show statistically significant associations with enrollment changes.

**Result:** PARTIALLY SUPPORTED  
Admissions metrics (applications, acceptance rate, yield rate) are statistically significant (p < 0.05). Affordability indicators show mixed results with small practical magnitudes.

---

## 📁 Repository Structure

```
thesis-enrollment-forecasting/
│
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── data/                              # Data directory
│   ├── README.md                      # Data documentation
│   └── .gitkeep                       # (IPEDS data not included due to size)
│
├── code/                              # Analysis code
│   ├── enrollment_forecasting_analysis.py  # Main analysis script
│   ├── data_preprocessing.py          # Data cleaning utilities
│   ├── forecasting_models.py          # Model implementations
│   ├── visualization.py               # Plotting functions
│   └── utils.py                       # Helper functions
│
├── results/                           # Analysis outputs
│   ├── tables/                        # CSV tables
│   │   ├── table_4.1_descriptive_statistics.csv
│   │   ├── table_4.2_annual_statistics.csv
│   │   ├── table_4.3_baseline_performance.csv
│   │   ├── table_4.4_arima_performance.csv
│   │   ├── table_4.5_ml_performance.csv
│   │   ├── table_4.6_regression_results.csv
│   │   └── table_4.7_model_comparison.csv
│   │
│   └── figures/                       # Visualizations
│       ├── fig1_enrollment_distribution.png
│       ├── fig2_enrollment_trend.png
│       ├── fig3_missing_data_pattern.png
│       ├── fig4_model_comparison_mae.png
│       └── fig5_model_comparison_mape.png
│
├── thesis/                            # LaTeX thesis documents
│   ├── main.tex                       # Main thesis file
│   ├── references.bib                 # Bibliography
│   ├── chapters/                      # Individual chapters
│   │   ├── chapter1_introduction.tex
│   │   ├── chapter2_literature.tex
│   │   ├── chapter3_methodology.tex
│   │   ├── chapter4_results.tex
│   │   └── chapter5_discussion.tex
│   │
│   └── figures/                       # Figures for LaTeX
│       └── (same as results/figures/)
│
├── presentation/                      # Pre-defense presentation
│   ├── PreDefense_RNU_Template_FINAL.pptx
│   └── SPEAKER_NOTES_GUIDE.txt
│
└── docs/                              # Additional documentation
    ├── COMPILATION.md                 # LaTeX compilation guide
    ├── METHODOLOGY.md                 # Detailed methodology
    └── DATA_DICTIONARY.md             # Variable descriptions
```

---

## 🔧 Installation

### Prerequisites

**Python Environment:**
```bash
# Python 3.8 or higher required
python --version
```

**LaTeX Distribution (for thesis compilation):**
- Windows: [MiKTeX](https://miktex.org/)
- macOS: [MacTeX](https://www.tug.org/mactex/)
- Linux: TeX Live
  ```bash
  sudo apt-get install texlive-full  # Ubuntu/Debian
  ```

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/thesis-enrollment-forecasting.git
   cd thesis-enrollment-forecasting
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Required Python Packages

```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
statsmodels>=0.13.0
scipy>=1.7.0
jupyter>=1.0.0
```

---

## 📊 Data

### Data Source

The analysis uses **IPEDS (Integrated Postsecondary Education Data System)** data, which is publicly available from the U.S. Department of Education:

**Download:** https://nces.ed.gov/ipeds/datacenter/

### Data Files Used

- **Institutional Characteristics (IC):** Institution type, control, size
- **Admissions (ADM):** Applications, admissions, acceptance rates
- **Enrollment (EF):** First-time enrollment (outcome variable)
- **Student Financial Aid (SFA):** Tuition, fees, financial aid
- **Human Resources (HR):** Faculty counts, student-faculty ratios

### Data Coverage

- **Time Period:** 2010-2021 (12 years)
- **Institutions:** 9,373 unique institutions
- **Observations:** 86,798 institution-year records
- **Complete Cases:** 67,513 (77.8% data completeness)

### Data Preparation

Due to file size limitations, raw IPEDS data is **not included** in this repository. 

**To replicate the analysis:**

1. Download IPEDS data from the link above
2. Place CSV files in `data/raw/`
3. Run preprocessing script:
   ```bash
   python code/data_preprocessing.py
   ```

**Alternative:** The analysis script includes a synthetic data generator that creates data matching the thesis characteristics for demonstration purposes.

---

## 🚀 Usage

### Quick Start

Run the complete analysis pipeline:

```bash
cd code
python enrollment_forecasting_analysis.py
```

This will:
- Load/generate data
- Compute descriptive statistics
- Run all forecasting models
- Perform panel regression
- Generate all tables (CSV) and figures (PNG)
- Save results to `results/` directory

### Step-by-Step Analysis

```python
# Load the analysis module
from enrollment_forecasting_analysis import *

# 1. Load data
df = load_and_prepare_data('data/processed/ipeds_panel.csv')

# 2. Descriptive statistics
desc_stats = compute_descriptive_statistics(df)

# 3. Baseline forecasting
baseline_results = baseline_forecasting(df)

# 4. Machine learning models
ml_results = ml_forecasting(df)

# 5. Panel regression
regression_results = panel_regression(df)

# 6. Model comparison
comparison = model_comparison(baseline_results, ml_results)
```

### Jupyter Notebook Analysis

Launch interactive analysis:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

---

## 🔬 Methodology

### Forecasting Models

1. **Naive Persistence Baseline**
   - Forecast: next_year = current_year
   - Benchmark for model evaluation
   
2. **Moving Average (k=3)**
   - Simple average of past 3 years
   - Smoothing approach

3. **ARIMA (AutoRegressive Integrated Moving Average)**
   - Aggregate time series model
   - National-level forecasting
   - Order: (1,1,1)

4. **Ridge Regression**
   - Linear model with L2 regularization
   - Panel data with institution features
   - Alpha: 1.0

5. **Random Forest**
   - Ensemble method (100 trees)
   - Nonlinear relationships
   - Panel data approach

### Validation Protocol

**Walk-Forward Validation:**
- Training Period: 2010-2017
- Test Period: 2018-2021 (4 years)
- Evaluation Metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Percentage Error (MAPE)

### Driver Analysis

**Panel Regression Specification:**
```
log(enrollment_t) = β₀ + β₁·log(enrollment_{t-1}) + β₂·log(applications_{t-1}) 
                    + β₃·acceptance_rate + β₄·yield_rate + β₅·log(net_price)
                    + β₆·log(grant_aid) + β₇·student_faculty_ratio 
                    + year_fixed_effects + ε
```

**Estimation:**
- OLS with robust standard errors
- Clustered at institution level
- Year fixed effects included

---

## 📈 Results

### Key Performance Metrics

| Model | Avg MAE | Avg MAPE (%) | Avg RMSE |
|-------|---------|--------------|----------|
| **Naive Persistence** | **39.43** | **0.64** | **113.06** |
| Ridge Regression | 40.33 | 0.70 | 113.11 |
| Random Forest | 41.21 | 1.22 | 115.11 |
| Moving Average | 45.87 | 0.99 | 127.48 |

✅ **Naive persistence outperforms all complex models**

### Panel Regression Results

| Variable | Coefficient | Std. Error | p-value |
|----------|------------|------------|---------|
| Lagged Enrollment (log) | 0.983 | 0.002 | <0.001 |
| Applications (log) | 0.012 | 0.003 | <0.001 |
| Acceptance Rate | -0.045 | 0.018 | 0.012 |
| Yield Rate | 0.028 | 0.011 | 0.011 |
| Net Price (log) | -0.015 | 0.006 | 0.012 |
| Grant Aid (log) | 0.008 | 0.004 | 0.045 |
| Student-Faculty Ratio | -0.002 | 0.001 | 0.021 |

**Model Fit:** R² = 0.976, Adjusted R² = 0.975

### COVID-19 Impact

All models experienced increased errors in 2020:
- Naive persistence MAE: 37.90 (2019) → 43.82 (2020) [+15.6%]
- Random Forest MAE: 40.19 (2019) → 46.05 (2020) [+14.6%]
- Random Forest MAPE: 1.03% (2019) → 1.92% (2020) [+86%]

**Conclusion:** Simple models proved more resilient to structural shocks.

---

## 📚 Documentation

### Thesis Chapters

**Chapter 1: Introduction**
- Research problem and motivation
- Research questions and hypotheses
- Thesis structure

**Chapter 2: Literature Review**
- Enrollment management theory
- Forecasting methods in education
- Previous empirical studies

**Chapter 3: Methodology**
- Data sources and variables
- Model specifications
- Validation protocol

**Chapter 4: Results and Analysis**
- Descriptive statistics
- Forecasting performance evaluation
- Driver analysis results
- Hypothesis testing

**Chapter 5: Discussion and Conclusion**
- Theoretical implications
- Practical recommendations
- Limitations and future research

### Compiling the Thesis

```bash
cd thesis
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Output: `main.pdf` (complete thesis)

### Alternative: Overleaf

Upload the `thesis/` directory to [Overleaf](https://www.overleaf.com) for online compilation.

---

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@bachelorthesis{Ashithosh Nithin 2026 enrollment,
  title={Forecasting University Enrollment Demand: IPEDS Data Analysis (2010-2021)},
  author={Ashithosh Nithin},
  year={2026},
  school={Riga Nordic University},
  type={Bachelor Thesis},
  address={Riga, Latvia}
}
```

**APA Style:**
```
Ashithosh Nithin. (2026). Forecasting university enrollment demand: IPEDS data analysis 
    (2010-2021) [Bachelor thesis, Riga Nordic University].
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Summary:**
- ✅ Free to use, modify, and distribute
- ✅ Commercial use permitted
- ✅ Attribution required
- ❌ No warranty provided

---

## 🙏 Acknowledgments

- **Supervisor:** Andrejs Bondarenko for invaluable guidance and feedback
- **Riga Nordic University** for institutional support
- **U.S. Department of Education** for making IPEDS data publicly available
- **Open-source community** for excellent tools (Python, LaTeX, scikit-learn)

### Libraries & Tools Used

- **Data Analysis:** pandas, numpy, scipy
- **Machine Learning:** scikit-learn
- **Statistical Modeling:** statsmodels
- **Visualization:** matplotlib, seaborn
- **Documentation:** LaTeX, Markdown

---

## 📧 Contact

**Author:** Ashithosh Nithin  
**Email:** Ashithosh Nithin  
**LinkedIn:**   
**GitHub:** [@Ashithosh-Nithin](https://github.com/Ashithosh-Nithin)

**Institution:** Riga Nordic University  
**Department:** Information Systems  
**Program:** Bachelor in information Systems

---

## 🔄 Project Status

- ✅ **Data Collection:** Complete
- ✅ **Analysis:** Complete
- ✅ **Thesis Writing:** Complete
- ✅ **Pre-Defense Presentation:** Complete
- 🔄 **Final Defense:** Scheduled for 13 january
- ⏳ **Publication:** Planned submission to [Journal Name]

---

## 🌟 Key Contributions

This thesis makes the following contributions to the literature:

1. **Empirical Evidence:** Demonstrates that enrollment persistence dominates forecasting accuracy using the most comprehensive U.S. dataset to date

2. **Methodological Rigor:** Applies walk-forward validation protocol superior to common single-split approaches

3. **Practical Guidance:** Provides clear recommendations for institutional researchers: embrace simple baselines

4. **Theoretical Insight:** Reframes enrollment dynamics as having persistent structural and marginal adjustment components

5. **Stress Test Analysis:** COVID-19 pandemic serves as natural experiment showing model robustness

---

## 🔗 Related Resources

### IPEDS Resources
- [IPEDS Data Center](https://nces.ed.gov/ipeds/datacenter/)
- [IPEDS Survey Components](https://nces.ed.gov/ipeds/report-your-data)
- [IPEDS Glossary](https://surveys.nces.ed.gov/ipeds/VisGlossaryAll.aspx)

### Forecasting Resources
- [Forecasting: Principles and Practice (Hyndman & Athanasopoulos)](https://otexts.com/fpp3/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [statsmodels Documentation](https://www.statsmodels.org/)

### Higher Education Analytics
- [National Student Clearinghouse Research Center](https://nscresearchcenter.org/)
- [EDUCAUSE Analytics](https://www.educause.edu/focus-areas-and-initiatives/policy-and-security/educause-policy/data-analytics)
- [AIR (Association for Institutional Research)](https://www.airweb.org/)

---

## 📝 Version History

- **v1.0.0** (January 2026) - Initial thesis submission
- **v0.9.0** (December 2025) - Pre-defense version
- **v0.5.0** (October 2025) - Methodology and preliminary results
- **v0.1.0** (August 2025) - Literature review and research design

---

## 🏆 Awards & Recognition

- [To be added upon defense/publication]

---

**Last Updated:** January 3, 2026  
**Repository:** https://github.com/yourusername/thesis-enrollment-forecasting  
**DOI:** [To be assigned upon publication]

---

<div align="center">

**Made with ❤️ for higher education analytics**

[⬆ Back to Top](#forecasting-university-enrollment-demand-ipeds-data-analysis-2010-2021)

</div>
