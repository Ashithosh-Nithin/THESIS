# Forecasting University Enrollment Demand in the United States

**Using IPEDS Administrative Panel Data (2010–2021)**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LaTeX](https://img.shields.io/badge/LaTeX-Thesis-blue)](https://www.latex-project.org/)
[![RNU](https://img.shields.io/badge/RNU-Bachelor%20Thesis-green)](https://www.rnu.lv/)

## 📖 Overview

This repository contains the LaTeX source code for a Bachelor's thesis examining enrollment forecasting in U.S. higher education institutions using IPEDS (Integrated Postsecondary Education Data System) administrative panel data from 2010 to 2021.

The thesis investigates:
- **Forecasting accuracy** of different time series and panel models for predicting first-time enrollment
- **Key determinants** of enrollment demand using econometric analysis
- **Baseline benchmarking** to assess whether sophisticated models outperform naive persistence forecasts

## 🎯 Research Questions

**RQ1:** Which forecasting method achieves the highest accuracy for first-time enrollment when evaluated using walk-forward validation?

**RQ2:** Which institutional and environmental factors are most strongly associated with changes in enrollment demand after controlling for persistence effects?

### Hypotheses

**H1:** Complex time series or machine learning models will outperform naive baseline forecasts by at least 10% in mean absolute error (MAE).

**H2:** Admissions funnel metrics (acceptance rate, yield rate) and affordability indicators (net price, aid coverage) will show statistically significant associations (p < 0.05) with enrollment changes.

## 📂 Repository Structure

```
├── chapter1/
│   └── chapter1.tex          # Introduction (10 sections, ~18-22 pages)
├── chapter2/
│   └── chapter2.tex          # Literature Review (8 sections, ~20-25 pages)
├── chapter3/
│   └── chapter3.tex          # Data and Methodology (12 sections, ~15-18 pages)
├── bibliography/
│   └── references.bib        # Complete bibliography (44 references)
├── README.md                 # This file
└── LICENSE                   # MIT License
```

## 📋 Chapter Contents

### Chapter 1: Introduction
- **1.1** Background and Rationale
- **1.2** Research Problem
- **1.3** Aim and Objectives (7 specific objectives)
- **1.4** Object and Subject of the Research
- **1.5** Research Questions and Hypotheses
- **1.6** Scope, Assumptions, and Limitations
- **1.7** Methods Overview
- **1.8** Expected Outcomes and Contributions
- **1.9** Scientific Novelty and Practical Significance
- **1.10** Structure of the Thesis

**Pages:** ~18-22 | **File size:** 22 KB

### Chapter 2: Literature Review and Theoretical Framework
- **2.1** Conceptualizing Enrollment Demand
- **2.2** Determinants of Enrollment Demand
  - 2.2.1 Conceptual Framework
- **2.3** Administrative Data Sources and Measurement
- **2.4** Forecasting Approaches for Enrollment Time Series
- **2.5** Panel Forecasting and Hierarchical Modeling
- **2.6** Driver Analysis and Limits of Inference
- **2.7** Forecast Evaluation, Validation, and Reproducibility
- **2.8** Summary and Implications

**Pages:** ~20-25 | **File size:** 32 KB | **Citations:** 40+

### Chapter 3: Data and Methodology
- **3.1** Research Design and Analytical Logic
- **3.2** Data Sources and Provenance
- **3.3** Study Period, Population, Analytic Sample
- **3.4** Outcome Definition and Predictor Operationalization
- **3.5** Data Processing Pipeline
  - 3.5.1 Missing-Data Strategy
- **3.6** Forecasting Methodology
  - 3.6.1 Baseline Forecasts (Naive, Moving Average)
  - 3.6.2 ARIMA Modelling on Aggregate Series
  - 3.6.3 Multivariate Panel Models (Ridge Regression, Random Forest)
- **3.7** Determinants and Explanatory Modelling
- **3.8** Validation Design and Performance Metrics
- **3.9** Robustness and Sensitivity Analyses
- **3.10** Ethical Considerations and Data Governance
- **3.11** Computational Environment and Reproducibility
- **3.12** Chapter Summary

**Pages:** ~15-18 | **File size:** 19 KB | **Equations:** 9

## 📊 Dataset

**Source:** IPEDS (Integrated Postsecondary Education Data System)  
**Years:** 2010–2021 (12 years)  
**Institutions:** U.S. degree-granting institutions  
**Observations:** 86,798 institution-year observations  
**Variables:** 235 variables across 5 topic files

### Key Variable Groups
- **Outcome:** First-time enrollment (`adm_number_enrolled_total`)
- **Persistence:** Lagged enrollment
- **Admissions Funnel:** Acceptance rate, yield rate, applications
- **Affordability:** Net price, grant aid, tuition
- **Capacity:** Student-faculty ratio, full-time faculty percentage

## 🔬 Methodology

### Forecasting Models
1. **Naive Baseline:** ŷ<sub>i,t</sub> = y<sub>i,t-1</sub>
2. **Moving Average:** k-year rolling window
3. **ARIMA:** Box-Jenkins methodology on aggregate series
4. **Ridge Regression:** L2-penalized linear model
5. **Random Forest:** Ensemble of decision trees

### Validation Protocol
- **Walk-forward validation** with expanding training window
- **Performance metrics:** MAE, RMSE, MAPE
- **Evaluation period:** 2018–2021

### Determinants Analysis
- **OLS regression** with year fixed effects
- **Robust standard errors** (HC3)
- **Associational analysis** (not causal inference)

## 🛠️ Technical Stack

**LaTeX Distribution:** TeX Live 2024  
**Document Class:** RNU Thesis Template  
**Bibliography:** BibLaTeX with IEEE style  
**Computational:** Python 3.11 (pandas, statsmodels, scikit-learn)

## 📚 Key References

This thesis builds on foundational work in:
- **College choice models:** Hossler & Gallagher (1987), Perna (2006)
- **Time series forecasting:** Box & Jenkins (2015), Hyndman & Athanasopoulos (2021)
- **Panel econometrics:** Wooldridge (2010), Baltagi (2005)
- **Machine learning:** Breiman (2001), Hastie et al. (2009)
- **Reproducible research:** Peng (2011), Gneiting & Katzfuss (2014)

**Total references:** 44 (see `bibliography/references.bib`)

## 🚀 Compilation Instructions

### Prerequisites
```bash
# Install LaTeX distribution
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install --cask mactex

# Windows
# Download and install MiKTeX from https://miktex.org/
```

### Building the Document

#### Method 1: Standard Compilation
```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

#### Method 2: Using LaTeXmk
```bash
latexmk -pdf main.tex
```

#### Method 3: Overleaf
1. Upload repository as ZIP to [Overleaf](https://www.overleaf.com/)
2. Set `main.tex` as main document
3. Click "Recompile"

## 📈 Expected Results

### Key Findings (Hypotheses Testing)
- **H1:** Expected to be REJECTED (naive baseline likely competitive)
- **H2:** Expected to be PARTIALLY SUPPORTED (some factors significant)

### Model Performance
- Enrollment shows **high persistence** (β ≈ 0.98, R² ≈ 0.98)
- Naive baseline likely achieves **MAE ≈ 39-41** students
- Complex models may **not substantially improve** over baseline

### Significant Determinants (Expected)
- Net price (negative association)
- Acceptance rate (U-shaped relationship)
- Applications received (positive association)
- Student-faculty ratio (institutional capacity constraint)

## 🎓 Academic Context

**Institution:** Riga Nordic University (RNU)  
**Program:** Bachelor of Science  
**Thesis Type:** Quantitative empirical research  
**Methodology:** Mixed methods (time series forecasting + panel regression)

## 📝 Citation

If you use this work, please cite:

```bibtex
@thesis{enrollment_forecasting_2025,
  author = {[Your Name]},
  title = {Forecasting University Enrollment Demand in the United States Using IPEDS Administrative Panel Data (2010--2021)},
  school = {Riga Nordic University},
  year = {2025},
  type = {Bachelor's Thesis}
}
```

## 🤝 Contributing

This is a thesis project and is not open for external contributions. However, feedback and suggestions are welcome via issues.

## 📧 Contact

**Author:** [Your Name]  
**Email:** [your.email@example.com]  
**Supervisor:** [Supervisor Name]  
**Institution:** Riga Nordic University

## 📄 License

This work is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Source:** National Center for Education Statistics (NCES) IPEDS
- **Data Access:** Urban Institute Education Data Portal
- **Thesis Template:** Riga Nordic University LaTeX Template
- **Supervisor:** [Supervisor Name] for guidance and feedback

## 📅 Timeline

- **Research Period:** 2010–2021 (data coverage)
- **Analysis Period:** 2024
- **Expected Defense:** [Month] 2025

## 🔍 Keywords

`enrollment forecasting` `higher education` `IPEDS` `time series analysis` `panel data` `machine learning` `econometrics` `administrative data` `walk-forward validation` `baseline benchmarking`

---

## 📊 Repository Statistics

- **Total Lines of LaTeX:** ~2,800 lines
- **Total Pages:** ~55-65 pages (Chapters 1-3)
- **References:** 44 academic sources
- **Equations:** 9 mathematical formulations
- **Tables:** 0 (in Chapters 1-3)
- **Figures:** 0 (in Chapters 1-3, conceptual framework placeholder)

---

**Status:** 🚧 Work in Progress - Chapters 1-3 Complete | Chapters 4-5 In Development

**Last Updated:** December 2024

---

⭐ **Star this repository if you find it useful!**
